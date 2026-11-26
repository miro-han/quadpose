"""Local HTTP + MJPEG + JSON-RPC backend for Team 7 Studio UI.

Uses only Python standard library (`http.server.ThreadingHTTPServer`) together
with existing project modules (`capture.camera_core`, `perception.segmentation.sam_session`),
requiring zero additional Python dependencies.
"""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from capture.camera_core import CaptureGroup, discover
from perception.segmentation.sam_session import ROOT, SamSession, visualize

STUDIO_DIR = Path(__file__).resolve().parent
DIST_DIR = STUDIO_DIR / "frontend" / "dist"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _detect_cuda() -> dict[str, Any]:
    try:
        import torch

        avail = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if avail else "CPU"
        return {"cuda": avail, "device": "cuda" if avail else "cpu", "name": name}
    except Exception:
        return {"cuda": False, "device": "cpu", "name": "CPU"}


def _encode_jpeg(frame: np.ndarray, max_width: int = 0, quality: int = 80) -> bytes:
    img = frame
    if max_width > 0 and img.shape[1] > max_width:
        scale = max_width / float(img.shape[1])
        img = cv2.resize(
            img,
            (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


class StudioState:
    """Thread-safe facade over CaptureGroup and SamSession."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.group = CaptureGroup()
        self.devices: dict[str, dict[str, Any]] = {}
        self.last_recording: Any = None
        self.recording_start_wall: float | None = None
        self.device_info = _detect_cuda()

        # SAM 2.1 interactive state
        self.sam = SamSession()
        self.sam_lock = threading.RLock()
        self.sam_busy = False
        self.sam_phase = "idle"  # idle | preparing | previewing | tracking | done | error | cancelled
        self.sam_message = "영상을 불러온 뒤 상자를 드래그해 물체를 지정하세요."
        self.sam_error = ""
        self.sam_progress = {"completed": 0, "total": 0, "elapsed_s": 0.0}
        self.sam_loaded_spec: tuple[str, int, int] | None = None
        self.sam_prepared_spec: tuple[str, int, int] | None = None
        self.sam_video_meta: dict[str, Any] = {}
        self.sam_raw_frame: np.ndarray | None = None
        self.sam_overlay_frame: np.ndarray | None = None
        self.sam_edges_frame: np.ndarray | None = None
        self.sam_frame_rev = 0
        self.sam_has_mask = False
        self.sam_mask_area_px = 0
        self.sam_output_folder: str | None = None

        # Downstream pipeline job runner (calibration, sync, edges, observations, 6DoF pose, benchmark)
        self.pipe_lock = threading.RLock()
        self.pipe_proc: subprocess.Popen[str] | None = None
        self.pipe_busy = False
        self.pipe_stage = ""
        self.pipe_phase = "idle"  # idle | running | done | error | cancelled
        self.pipe_message = "실행할 파이프라인 단계를 선택하세요."
        self.pipe_command = ""
        self.pipe_logs: list[str] = []
        self.pipe_started: float | None = None
        self.pipe_elapsed_s = 0.0
        self.pipe_output_dir: str | None = None
        self.pipe_preview_path: str | None = None
        self.pipe_result_summary: dict[str, Any] | None = None
        self.pipe_rev = 0

    # ---- System & Pipeline helpers ----

    def system_info(self) -> dict[str, Any]:
        return {
            "root": str(ROOT),
            "default_recordings_dir": str(ROOT / "recordings"),
            "default_outputs_dir": str(ROOT / "outputs" / "segmentation"),
            "compute": self.device_info,
            "videos": self.list_videos(),
        }

    def list_videos(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        search_dirs = [
            ("sample", ROOT / "data" / "sample_videos"),
            ("recording", ROOT / "recordings"),
            ("output", ROOT / "outputs" / "segmentation"),
        ]
        for category, base in search_dirs:
            if not base.exists():
                continue
            for path in sorted(base.rglob("*"), key=lambda p: p.stat().st_mtime if p.is_file() else 0, reverse=True):
                if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                rel = str(path.relative_to(ROOT))
                results.append(
                    {
                        "path": resolved,
                        "label": rel.replace("\\", "/"),
                        "name": path.name,
                        "category": category,
                        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                    }
                )
                if len(results) >= 40:
                    return results
        return results

    def list_sessions(self) -> dict[str, Any]:
        recordings = []
        rec_root = ROOT / "recordings"
        if rec_root.exists():
            for folder in sorted(rec_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not folder.is_dir():
                    continue
                meta_path = folder / "session.json"
                meta: dict[str, Any] = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}
                recordings.append(
                    {
                        "name": folder.name,
                        "path": str(folder.resolve()),
                        "status": meta.get("status", "unknown"),
                        "note": meta.get("note", ""),
                        "created_at": meta.get("created_at", ""),
                        "cameras": len(meta.get("cameras", {})),
                        "written_frames": meta.get("written_frames", {}),
                    }
                )

        segmentations = []
        seg_root = ROOT / "outputs" / "segmentation"
        if seg_root.exists():
            for folder in sorted(seg_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not folder.is_dir():
                    continue
                meta_path = folder / "session.json"
                meta = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}
                segmentations.append(
                    {
                        "name": folder.name,
                        "path": str(folder.resolve()),
                        "status": meta.get("status", "unknown"),
                        "source": Path(meta.get("source", "")).name if meta.get("source") else "",
                        "source_path": str(meta.get("source", "")),
                        "frame_count": meta.get("frame_count", 0),
                        "written_frames": meta.get("written_frames", 0),
                        "device": meta.get("device", ""),
                        "processing_seconds": round(float(meta.get("processing_seconds", 0.0)), 2),
                    }
                )

        edges = []
        edge_root = ROOT / "outputs" / "edges"
        if edge_root.exists():
            for folder in sorted(edge_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not folder.is_dir():
                    continue
                meta_path = folder / "summary.json"
                meta = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}
                edges.append(
                    {
                        "name": folder.name,
                        "path": str(folder.resolve()),
                        "camera_id": meta.get("camera_id", "camera_1"),
                        "source": Path(meta.get("source", "")).name if meta.get("source") else "",
                        "frames": meta.get("frames", 0),
                    }
                )

        pipeline_runs = []
        for kind, sub in (
            ("calibration", "calibration"),
            ("sync", "sync"),
            ("observations", "observations"),
            ("pose", "pose"),
            ("benchmark", "benchmarks"),
        ):
            base = ROOT / "outputs" / sub
            if not base.exists():
                continue
            for item in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                pipeline_runs.append(
                    {
                        "kind": kind,
                        "name": item.name,
                        "path": str(item.resolve()),
                        "is_dir": item.is_dir(),
                        "mtime": datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
        pipeline_runs.sort(key=lambda r: r["mtime"], reverse=True)

        return {
            "recordings": recordings[:20],
            "segmentations": segmentations[:20],
            "edges": edges[:20],
            "pipeline_runs": pipeline_runs[:30],
        }

    # ---- Camera Studio ----

    def camera_scan(self, backend: str = "DirectShow") -> dict[str, Any]:
        cv_backend = cv2.CAP_DSHOW if backend == "DirectShow" else cv2.CAP_MSMF
        found = discover(cv_backend)
        with self.lock:
            self.devices = {f"[{d['index']}] {d['name']}": d for d in found}
        items = [{"key": f"[{d['index']}] {d['name']}", **d} for d in found]
        return {"backend": backend, "devices": items}

    def camera_connect(
        self, slots: dict[str, str], resolution: str = "1280x720", fps: int = 30
    ) -> dict[str, Any]:
        with self.lock:
            if self.group.workers:
                self.group.disconnect()
                deadline = time.monotonic() + 3.0
                while not self.group.clear_stopped() and time.monotonic() < deadline:
                    time.sleep(0.05)
            selected: dict[int, dict[str, Any]] = {}
            for slot_str, dev_key in slots.items():
                if not dev_key or dev_key == "none":
                    continue
                if dev_key not in self.devices:
                    raise ValueError(f"알 수 없는 카메라 장치입니다: {dev_key}")
                selected[int(slot_str)] = self.devices[dev_key]
            w, h = map(int, resolution.split("x"))
            self.group.connect(selected, (w, h), int(fps))
        return self.camera_status()

    def camera_disconnect(self) -> dict[str, Any]:
        with self.lock:
            self.group.disconnect()
            deadline = time.monotonic() + 4.0
            while not self.group.clear_stopped() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.recording_start_wall = None
        return self.camera_status()

    def camera_record_start(self, directory: str, fps: int = 30, note: str = "") -> dict[str, Any]:
        target_dir = directory.strip() or str(ROOT / "recordings")
        with self.lock:
            rec = self.group.start_recording(target_dir, int(fps), note)
            self.last_recording = rec
            self.recording_start_wall = time.monotonic()
        return self.camera_status()

    def camera_record_stop(self) -> dict[str, Any]:
        with self.lock:
            rec = self.group.stop_recording()
            if rec is not None:
                self.last_recording = rec
            self.recording_start_wall = None
        return self.camera_status()

    def camera_status(self) -> dict[str, Any]:
        now_ns = time.perf_counter_ns()
        with self.lock:
            self.group.clear_stopped()
            rec = self.group.recording
            if rec is not None and rec.done.is_set():
                self.group.stop_recording()
                self.last_recording = rec
                rec = None
                self.recording_start_wall = None

            slots = []
            for slot in range(1, 5):
                worker = self.group.workers.get(slot)
                if worker is None:
                    slots.append(
                        {
                            "slot": slot,
                            "connected": False,
                            "status": "대기",
                            "error": "",
                            "fps": 0.0,
                            "size_px": None,
                            "stalled": False,
                            "written": 0,
                            "dropped": 0,
                            "device_name": "",
                        }
                    )
                else:
                    frame, seq, stamp, rate, state, err = worker.snapshot()
                    stalled = bool(stamp and (now_ns - stamp > 2_000_000_000))
                    slots.append(
                        {
                            "slot": slot,
                            "connected": True,
                            "has_frame": frame is not None,
                            "sequence": seq,
                            "status": "수신 지연" if stalled else state,
                            "error": err,
                            "fps": round(float(rate), 1),
                            "size_px": worker.actual.get("size_px"),
                            "stalled": stalled,
                            "written": rec.counts.get(slot, 0) if rec else 0,
                            "dropped": rec.dropped.get(slot, 0) if rec else 0,
                            "device_name": worker.device.get("name", ""),
                        }
                    )

            active_rec = rec is not None
            last = rec or self.last_recording
            rec_info = {
                "active": active_rec,
                "ready": self.group.ready(),
                "elapsed_s": round(time.monotonic() - self.recording_start_wall, 1)
                if active_rec and self.recording_start_wall
                else 0.0,
                "folder": str(last.folder) if last else "",
                "folder_name": last.folder.name if last else "",
                "total_written": sum(last.counts.values()) if last else 0,
                "total_dropped": sum(last.dropped.values()) if last else 0,
                "error": last.error if last else "",
                "done": bool(last.done.is_set()) if last else False,
            }
            return {
                "connected": bool(self.group.workers),
                "slots": slots,
                "recording": rec_info,
            }

    # ---- SAM 2.1 Interactive Tracking ----

    def sam_load_frame(self, video_path: str, start: int = 0, length: int = 30) -> dict[str, Any]:
        with self.sam_lock:
            if self.sam_busy:
                raise RuntimeError("현재 SAM 2.1 작업이 실행 중입니다.")
        start_i = int(start)
        length_i = int(length)
        if start_i < 0 or not (1 <= length_i <= 120):
            raise ValueError("시작 프레임은 0 이상, 처리 길이는 1~120 프레임이어야 합니다.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if total <= 0 or start_i >= total:
                raise ValueError(f"시작 프레임({start_i})이 영상 길이({total})를 벗어났습니다.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_i)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("선택한 프레임을 읽지 못했습니다.")
        finally:
            cap.release()

        clipped_len = min(length_i, total - start_i)
        with self.sam_lock:
            self.sam_loaded_spec = (video_path, start_i, clipped_len)
            self.sam_raw_frame = frame
            self.sam_overlay_frame = None
            self.sam_edges_frame = None
            self.sam_has_mask = False
            self.sam_mask_area_px = 0
            self.sam_frame_rev += 1
            self.sam_phase = "idle"
            self.sam_error = ""
            self.sam_message = f"프레임 {start_i} 로드 완료 ({frame.shape[1]}×{frame.shape[0]}). 상자를 드래그하세요."
            self.sam_video_meta = {
                "path": video_path,
                "total_frames": total,
                "fps": round(fps, 2),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "start": start_i,
                "length": clipped_len,
            }
        return self.sam_status()

    def sam_preview(
        self, box: list[float], points: list[list[float]] | None = None, labels: list[int] | None = None
    ) -> dict[str, Any]:
        with self.sam_lock:
            if self.sam_busy:
                raise RuntimeError("현재 다른 SAM 2.1 작업이 진행 중입니다.")
            if self.sam_loaded_spec is None or self.sam_raw_frame is None:
                raise ValueError("먼저 영상을 불러오세요.")
            if not box or len(box) != 4:
                raise ValueError("먼저 대상 물체에 박스를 그려주세요.")
            spec = self.sam_loaded_spec
            raw_frame = self.sam_raw_frame.copy()
            self.sam_busy = True
            self.sam_phase = "previewing"
            self.sam_error = ""
            self.sam_message = "SAM 2.1 마스크를 계산하는 중…"

        pts = points or []
        lbls = labels or []

        def worker() -> None:
            try:
                def notify(msg: str) -> None:
                    with self.sam_lock:
                        self.sam_message = msg

                if self.sam_prepared_spec != spec or self.sam.state is None:
                    with self.sam_lock:
                        self.sam_phase = "preparing"
                    self.sam.prepare(spec[0], spec[1], spec[2], notify)
                    with self.sam_lock:
                        self.sam_prepared_spec = spec
                        self.sam_phase = "previewing"

                notify("프롬프트로 마스크 생성 중…")
                mask = self.sam.preview(box, pts, lbls)
                overlay, edges = visualize(raw_frame, mask)
                edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                area = int(mask.sum())
                with self.sam_lock:
                    self.sam_overlay_frame = overlay
                    self.sam_edges_frame = edges_bgr
                    self.sam_has_mask = True
                    self.sam_mask_area_px = area
                    self.sam_frame_rev += 1
                    self.sam_phase = "ready"
                    self.sam_message = f"마스크 확인 완료 (면적 {area:,} px). 포함/제외 클릭으로 보정하거나 추적을 시작하세요."
                    self.sam_output_folder = str(self.sam.folder) if self.sam.folder else None
            except Exception as exc:
                with self.sam_lock:
                    self.sam_phase = "error"
                    self.sam_error = str(exc)
                    self.sam_message = f"오류: {exc}"
            finally:
                with self.sam_lock:
                    self.sam_busy = False

        threading.Thread(target=worker, daemon=True, name="sam-preview").start()
        return self.sam_status()

    def sam_track(self) -> dict[str, Any]:
        with self.sam_lock:
            if self.sam_busy:
                raise RuntimeError("현재 다른 SAM 2.1 작업이 진행 중입니다.")
            if not self.sam_has_mask or self.sam.state is None:
                raise ValueError("먼저 '마스크 확인'으로 초기 마스크를 생성하세요.")
            total = int(self.sam.metadata.get("frame_count", 0))
            self.sam_busy = True
            self.sam_phase = "tracking"
            self.sam_error = ""
            self.sam_progress = {"completed": 0, "total": total, "elapsed_s": 0.0}
            self.sam_message = "선택 구간 전체 추적을 시작합니다…"

        def worker() -> None:
            started = time.perf_counter()
            try:
                def notify(msg: str) -> None:
                    with self.sam_lock:
                        self.sam_message = msg

                def show(frame: np.ndarray, mask: np.ndarray, index: int) -> None:
                    overlay, edges = visualize(frame, mask)
                    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    with self.sam_lock:
                        self.sam_raw_frame = frame
                        self.sam_overlay_frame = overlay
                        self.sam_edges_frame = edges_bgr
                        self.sam_mask_area_px = int(mask.sum())
                        self.sam_frame_rev += 1
                        self.sam_progress = {
                            "completed": index + 1,
                            "total": total,
                            "elapsed_s": round(time.perf_counter() - started, 1),
                        }

                folder = self.sam.track(notify, show)
                with self.sam_lock:
                    cancelled = self.sam.stop.is_set()
                    self.sam_phase = "cancelled" if cancelled else "done"
                    self.sam_output_folder = str(folder)
                    self.sam_message = (
                        f"추적이 중단되었습니다. 부분 결과: {folder.name}"
                        if cancelled
                        else f"추적 완료! 결과 저장: {folder.name}"
                    )
            except Exception as exc:
                with self.sam_lock:
                    self.sam_phase = "error"
                    self.sam_error = str(exc)
                    self.sam_message = f"추적 오류: {exc}"
            finally:
                with self.sam_lock:
                    self.sam_busy = False

        threading.Thread(target=worker, daemon=True, name="sam-track").start()
        return self.sam_status()

    def sam_cancel(self) -> dict[str, Any]:
        self.sam.stop.set()
        with self.sam_lock:
            self.sam_message = "작업 중단을 요청했습니다…"
        return self.sam_status()

    def sam_status(self) -> dict[str, Any]:
        with self.sam_lock:
            return {
                "busy": self.sam_busy,
                "phase": self.sam_phase,
                "message": self.sam_message,
                "error": self.sam_error,
                "progress": dict(self.sam_progress),
                "video": dict(self.sam_video_meta),
                "has_frame": self.sam_raw_frame is not None,
                "has_mask": self.sam_has_mask,
                "mask_area_px": self.sam_mask_area_px,
                "frame_rev": self.sam_frame_rev,
                "output_folder": self.sam_output_folder,
                "compute": self.device_info,
            }

    def get_sam_frame_jpeg(self, mode: str = "overlay") -> bytes | None:
        with self.sam_lock:
            if mode == "overlay" and self.sam_overlay_frame is not None:
                img = self.sam_overlay_frame
            elif mode == "edges" and self.sam_edges_frame is not None:
                img = self.sam_edges_frame
            elif self.sam_raw_frame is not None:
                img = self.sam_raw_frame
            else:
                return None
            return _encode_jpeg(img, max_width=1280, quality=85)

    def open_folder(self, path_str: str) -> dict[str, Any]:
        target = Path(path_str) if path_str else (ROOT / "recordings")
        if target.is_file():
            target = target.parent
        target.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(target))
        return {"opened": str(target)}

    # ---- Downstream Pipeline Runner ----

    def _resolve_input_path(self, raw: str, label: str) -> Path:
        text = raw.strip()
        if not text:
            raise ValueError(f"{label} 경로를 입력하세요.")
        p = Path(text)
        if not p.is_absolute():
            p = ROOT / p
        p = p.resolve()
        if not p.exists():
            raise ValueError(f"{label} 경로를 찾을 수 없습니다: {p}")
        return p

    def _summarize_pipeline_output(self, stage: str, output_dir: Path) -> tuple[str | None, dict[str, Any] | None]:
        preview: str | None = None
        summary: dict[str, Any] = {"output": str(output_dir)}

        # Look for overlay image or extracted frame
        for candidate in (
            output_dir / "overlay.jpg",
            output_dir / "frame" / "overlay.jpg",
            output_dir if output_dir.is_file() and output_dir.suffix.lower() in {".png", ".jpg", ".jpeg"} else None,
        ):
            if candidate is not None and candidate.is_file():
                preview = str(candidate)
                break

        # Parse stage-specific JSON/CSV outputs
        if stage in ("observations_extract", "observations_demo"):
            obs_path = (
                output_dir / "frame" / "observations.json"
                if (output_dir / "frame" / "observations.json").is_file()
                else output_dir / "observations.json"
            )
            if obs_path.is_file():
                data = json.loads(obs_path.read_text(encoding="utf-8-sig"))
                segs = data.get("segments", [])
                corners = data.get("corners", [])
                accepted_segs = [s for s in segs if s.get("reason") is None]
                summary.update(
                    {
                        "camera_id": data.get("camera_id"),
                        "source_frame": data.get("source_frame"),
                        "segments_accepted": len(accepted_segs),
                        "segments_total": len(segs),
                        "corners": len(corners),
                        "pixel_space": data.get("pixel_space"),
                    }
                )
        elif stage == "pose_fuse_group" and output_dir.is_file():
            data = json.loads(output_dir.read_text(encoding="utf-8-sig"))
            summary.update(
                {
                    "valid": data.get("valid"),
                    "method": data.get("method"),
                    "group_id": data.get("group_id"),
                    "excluded_camera_ids": data.get("excluded_camera_ids", []),
                    "failure_reason": data.get("failure_reason"),
                }
            )
        elif stage == "edges_lsd" and (output_dir / "summary.json").is_file():
            data = json.loads((output_dir / "summary.json").read_text(encoding="utf-8-sig"))
            summary.update(
                {
                    "frames": data.get("frames"),
                    "camera_id": data.get("camera_id"),
                    "mean_candidates": round(float(data.get("mean_candidates_per_frame", 0.0)), 1),
                }
            )
        elif stage == "pose_benchmark" and (output_dir / "summary.csv").is_file():
            rows = []
            with (output_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("layout") == "pooled" and r.get("regime") == "pooled" and r.get("window") == "all_frames":
                        rows.append(
                            {
                                "method": r.get("method"),
                                "success_pct": round(float(r.get("tracking_success_pct") or 0), 2),
                                "edge_f1_pct": round(float(r.get("edge_f1_pct") or 0), 2),
                                "false_valid_pct": round(float(r.get("false_valid_pct") or 0), 2),
                            }
                        )
            summary["methods"] = rows

        return preview, summary

    def pipeline_run(self, stage: str, params: dict[str, Any]) -> dict[str, Any]:
        with self.pipe_lock:
            if self.pipe_busy:
                raise RuntimeError("현재 다른 파이프라인 작업이 실행 중입니다.")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        py = sys.executable
        cmd: list[str]
        output_path: Path

        if stage == "rig_extract_frame":
            session_dir = self._resolve_input_path(str(params.get("session_dir", "")), "녹화 세션")
            slot = int(params.get("slot", 1))
            frame_idx = int(params.get("frame", 0))
            out_dir = ROOT / "outputs" / "calibration"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{session_dir.name}_slot{slot}_f{frame_idx:04d}_{stamp}.png"
            cmd = [
                py, "-B", "-m", "rig.extract_frame",
                "--session-dir", str(session_dir),
                "--slot", str(slot),
                "--frame", str(frame_idx),
                "--output", str(output_path),
            ]
        elif stage == "rig_calibrate":
            cfg = self._resolve_input_path(str(params.get("config", "")), "보정 설정(rig_input.json)")
            out_dir = ROOT / "outputs" / "calibration"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"rig_{stamp}.json"
            cmd = [py, "-B", "-m", "rig.calibrate", "--config", str(cfg), "--output", str(output_path)]
        elif stage == "rig_sync_flash":
            cfg = self._resolve_input_path(str(params.get("config", "")), "플래시 설정(flash_input.json)")
            out_dir = ROOT / "outputs" / "sync"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"flash_{stamp}.json"
            cmd = [py, "-B", "-m", "rig.synchronize", "flash", "--config", str(cfg), "--output", str(output_path)]
        elif stage == "rig_sync_align":
            cfg = self._resolve_input_path(str(params.get("config", "")), "정렬 설정(align_input.json)")
            out_dir = ROOT / "outputs" / "sync"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"alignment_{stamp}.json"
            cmd = [py, "-B", "-m", "rig.synchronize", "align", "--config", str(cfg), "--output", str(output_path)]
        elif stage == "edges_lsd":
            video = self._resolve_input_path(str(params.get("video", "")), "영상")
            camera_id = str(params.get("camera_id", "camera_1")).strip() or "camera_1"
            output_path = ROOT / "outputs" / "edges" / f"lsd_{video.stem}_{stamp}"
            cmd = [
                py, "-B", "-m", "perception.edges.check_segments",
                "--video", str(video),
                "--camera-id", camera_id,
                "--output", str(output_path),
            ]
        elif stage == "observations_demo":
            output_path = ROOT / "outputs" / "observations" / f"demo_single_{stamp}"
            cmd = [py, "-B", "-m", "perception.observations.demo_single", "--output", str(output_path)]
        elif stage == "observations_extract":
            video = self._resolve_input_path(str(params.get("video", "")), "영상")
            session = self._resolve_input_path(str(params.get("session", "")), "SAM 2.1 세션")
            lsd = self._resolve_input_path(str(params.get("lsd", "")), "LSD 폴더")
            frame_idx = int(params.get("frame", 0))
            camera_id = str(params.get("camera_id", "camera_1")).strip() or "camera_1"
            output_path = ROOT / "outputs" / "observations" / f"obs_{video.stem}_f{frame_idx}_{stamp}"
            cmd = [
                py, "-B", "-m", "perception.observations.adapter",
                "--video", str(video),
                "--session", str(session),
                "--lsd", str(lsd),
                "--frame", str(frame_idx),
                "--camera-id", camera_id,
                "--output", str(output_path),
            ]
            rig_raw = str(params.get("rig", "")).strip()
            if rig_raw:
                cmd.extend(["--rig", str(self._resolve_input_path(rig_raw, "rig.json"))])
            prior_raw = str(params.get("prior", "")).strip()
            if prior_raw:
                cmd.extend(["--prior", str(self._resolve_input_path(prior_raw, "prior.json"))])
        elif stage == "pose_fuse_group":
            cfg = self._resolve_input_path(str(params.get("config", "")), "그룹 설정(group.json)")
            out_dir = ROOT / "outputs" / "pose"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"pose_{stamp}.json"
            cmd = [py, "-B", "-m", "perception.observations.fuse_group", "--config", str(cfg), "--output", str(output_path)]
        elif stage == "pose_benchmark":
            suite = str(params.get("suite", "selection")).strip()
            if suite not in ("selection", "assignment", "primary"):
                raise ValueError(f"지원하지 않는 벤치마크 suite입니다: {suite}")
            smoke = bool(params.get("smoke", True))
            output_path = ROOT / "outputs" / "benchmarks" / f"{suite}_{'smoke_' if smoke else ''}{stamp}"
            cmd = [
                py, "-B", "-m", "evaluation.resource_limits",
                "--threads", "1", "--memory-gib", "3",
                "evaluation.pose_benchmark.benchmark",
                "--suite", suite,
                "--output", str(output_path),
            ]
            if smoke:
                cmd.append("--smoke")
        else:
            raise ValueError(f"알 수 없는 파이프라인 단계입니다: {stage}")

        cmd_str = " ".join(cmd)
        with self.pipe_lock:
            self.pipe_busy = True
            self.pipe_stage = stage
            self.pipe_phase = "running"
            self.pipe_message = f"{stage} 실행 중…"
            self.pipe_command = cmd_str
            self.pipe_logs = [f"$ {cmd_str}"]
            self.pipe_started = time.perf_counter()
            self.pipe_elapsed_s = 0.0
            self.pipe_output_dir = str(output_path.parent if output_path.suffix else output_path)
            self.pipe_preview_path = None
            self.pipe_result_summary = None
            self.pipe_rev += 1

        def worker() -> None:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                with self.pipe_lock:
                    self.pipe_proc = proc

                assert proc.stdout is not None
                for line in proc.stdout:
                    text = line.rstrip("\r\n")
                    if not text:
                        continue
                    with self.pipe_lock:
                        self.pipe_logs.append(text)
                        if len(self.pipe_logs) > 200:
                            self.pipe_logs = self.pipe_logs[-200:]
                        if self.pipe_started is not None:
                            self.pipe_elapsed_s = round(time.perf_counter() - self.pipe_started, 1)

                rc = proc.wait()
                elapsed = round(time.perf_counter() - (self.pipe_started or time.perf_counter()), 1)
                with self.pipe_lock:
                    self.pipe_proc = None
                    self.pipe_elapsed_s = elapsed
                    if self.pipe_phase == "cancelled":
                        self.pipe_message = f"{stage} 작업이 중단되었습니다."
                    elif rc == 0:
                        preview, summary = self._summarize_pipeline_output(stage, output_path)
                        self.pipe_preview_path = preview
                        self.pipe_result_summary = summary
                        self.pipe_phase = "done"
                        self.pipe_message = f"{stage} 완료 ({elapsed}s)"
                    else:
                        self.pipe_phase = "error"
                        self.pipe_message = f"{stage} 실패 (종료 코드 {rc})"
                    self.pipe_rev += 1
            except Exception as exc:
                with self.pipe_lock:
                    self.pipe_proc = None
                    self.pipe_phase = "error"
                    self.pipe_message = f"오류: {exc}"
                    self.pipe_logs.append(f"[ERROR] {exc}")
                    self.pipe_rev += 1
            finally:
                with self.pipe_lock:
                    self.pipe_busy = False

        threading.Thread(target=worker, daemon=True, name=f"pipe-{stage}").start()
        return self.pipeline_status()

    def pipeline_cancel(self) -> dict[str, Any]:
        with self.pipe_lock:
            proc = self.pipe_proc
            if proc is not None and proc.poll() is None:
                self.pipe_phase = "cancelled"
                self.pipe_message = "작업 중단을 요청했습니다…"
                try:
                    proc.terminate()
                except Exception:
                    pass
        return self.pipeline_status()

    def pipeline_status(self) -> dict[str, Any]:
        with self.pipe_lock:
            elapsed = (
                round(time.perf_counter() - self.pipe_started, 1)
                if self.pipe_busy and self.pipe_started is not None
                else self.pipe_elapsed_s
            )
            return {
                "busy": self.pipe_busy,
                "stage": self.pipe_stage,
                "phase": self.pipe_phase,
                "message": self.pipe_message,
                "command": self.pipe_command,
                "logs": list(self.pipe_logs[-80:]),
                "elapsed_s": elapsed,
                "output_dir": self.pipe_output_dir,
                "has_preview": bool(self.pipe_preview_path and Path(self.pipe_preview_path).is_file()),
                "summary": self.pipe_result_summary,
                "rev": self.pipe_rev,
            }

    def get_pipeline_preview_jpeg(self) -> bytes | None:
        with self.pipe_lock:
            path_str = self.pipe_preview_path
        if not path_str:
            return None
        p = Path(path_str)
        if not p.is_file():
            return None
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return _encode_jpeg(img, max_width=1280, quality=85)

    # ---- JSON-RPC Dispatcher ----

    def dispatch(self, method: str, params: dict[str, Any]) -> Any:
        handlers = {
            "system.info": lambda p: self.system_info(),
            "system.open_folder": lambda p: self.open_folder(str(p.get("path", ""))),
            "pipeline.sessions": lambda p: self.list_sessions(),
            "pipeline.run": lambda p: self.pipeline_run(str(p.get("stage", "")), dict(p.get("params", {}))),
            "pipeline.status": lambda p: self.pipeline_status(),
            "pipeline.cancel": lambda p: self.pipeline_cancel(),
            "camera.scan": lambda p: self.camera_scan(str(p.get("backend", "DirectShow"))),
            "camera.connect": lambda p: self.camera_connect(
                dict(p.get("slots", {})),
                str(p.get("resolution", "1280x720")),
                int(p.get("fps", 30)),
            ),
            "camera.disconnect": lambda p: self.camera_disconnect(),
            "camera.status": lambda p: self.camera_status(),
            "camera.record_start": lambda p: self.camera_record_start(
                str(p.get("directory", "")),
                int(p.get("fps", 30)),
                str(p.get("note", "")),
            ),
            "camera.record_stop": lambda p: self.camera_record_stop(),
            "sam.videos": lambda p: {"videos": self.list_videos()},
            "sam.load_frame": lambda p: self.sam_load_frame(
                str(p.get("video_path", "")),
                int(p.get("start", 0)),
                int(p.get("length", 30)),
            ),
            "sam.preview": lambda p: self.sam_preview(
                list(p.get("box", [])),
                list(p.get("points", [])),
                list(p.get("labels", [])),
            ),
            "sam.track": lambda p: self.sam_track(),
            "sam.cancel": lambda p: self.sam_cancel(),
            "sam.status": lambda p: self.sam_status(),
        }
        if method not in handlers:
            raise ValueError(f"Unknown RPC method: {method}")
        return handlers[method](params)


STATE = StudioState()


class StudioHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep console output clean; only log errors if needed.
        return

    def _allowed_origin(self) -> str | None:
        origin = (self.headers.get("Origin") or "").strip()
        if not origin:
            return None
        parsed = urlparse(origin)
        if parsed.scheme in ("http", "https") and parsed.hostname in ("127.0.0.1", "localhost"):
            return origin
        if parsed.scheme == "tauri" and parsed.hostname == "localhost":
            return origin
        return None

    def _send_cors(self) -> None:
        allowed = self._allowed_origin()
        if allowed:
            self.send_header("Access-Control-Allow-Origin", allowed)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        origin = (self.headers.get("Origin") or "").strip()
        if origin and self._allowed_origin() is None:
            self._send_json(
                {
                    "schema_version": 1,
                    "id": "",
                    "result": None,
                    "error": {"code": "FORBIDDEN_ORIGIN", "message": "Cross-origin request blocked"},
                },
                status=403,
            )
            return

        parsed = urlparse(self.path)
        if parsed.path != "/api/rpc":
            self._send_json({"error": "Not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json(
                {
                    "schema_version": 1,
                    "id": "",
                    "result": None,
                    "error": {"code": "INVALID_JSON", "message": "Invalid JSON body"},
                },
                status=400,
            )
            return

        req_id = str(req.get("id", ""))
        method = str(req.get("method", ""))
        params = req.get("params") or {}
        try:
            result = STATE.dispatch(method, params)
            self._send_json(
                {"schema_version": 1, "id": req_id, "result": result, "error": None}
            )
        except Exception as exc:
            self._send_json(
                {
                    "schema_version": 1,
                    "id": req_id,
                    "result": None,
                    "error": {"code": "RPC_ERROR", "message": str(exc)},
                }
            )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._send_json({"ok": True, "compute": STATE.device_info})
            return

        if path.startswith("/api/cameras/") and path.endswith("/stream"):
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[2].isdigit():
                self._serve_mjpeg(int(parts[2]))
                return

        if path == "/api/sam/frame.jpg":
            qs = parse_qs(parsed.query)
            mode = qs.get("mode", ["overlay"])[0]
            jpeg = STATE.get_sam_frame_jpeg(mode)
            if jpeg is None:
                self.send_response(HTTPStatus.NOT_FOUND)
                self._send_cors()
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(HTTPStatus.OK)
            self._send_cors()
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)
            return

        if path == "/api/pipeline/preview.jpg":
            jpeg = STATE.get_pipeline_preview_jpeg()
            if jpeg is None:
                self.send_response(HTTPStatus.NOT_FOUND)
                self._send_cors()
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(HTTPStatus.OK)
            self._send_cors()
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)
            return

        self._serve_static(path)

    def _serve_mjpeg(self, slot: int) -> None:
        boundary = b"frame"
        self.send_response(HTTPStatus.OK)
        self._send_cors()
        self.send_header("Connection", "close")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.close_connection = True

        last_seq = -1
        try:
            while True:
                worker = STATE.group.workers.get(slot)
                if worker is None:
                    break
                frame, seq, _, _, _, _ = worker.snapshot()
                if frame is not None and seq != last_seq:
                    last_seq = seq
                    jpeg = _encode_jpeg(frame, max_width=640, quality=78)
                    header = (
                        b"--"
                        + boundary
                        + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode("ascii")
                        + b"\r\n\r\n"
                    )
                    self.wfile.write(header)
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                time.sleep(0.05)  # ~20 fps preview cap to avoid GIL contention during recording
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

    def _serve_static(self, req_path: str) -> None:
        if not DIST_DIR.exists():
            self._send_json(
                {
                    "message": "Frontend not built yet. Run `npm.cmd run build` inside studio/frontend.",
                    "dist": str(DIST_DIR),
                },
                status=503,
            )
            return

        clean = req_path.lstrip("/") or "index.html"
        target = (DIST_DIR / clean).resolve()
        try:
            target.relative_to(DIST_DIR.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return

        if not target.is_file():
            target = DIST_DIR / "index.html"
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        data = target.read_bytes()
        mime, _ = mimetypes.guess_type(str(target))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Team 7 Studio local server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    cv2.setNumThreads(2)
    server = ThreadingHTTPServer((args.host, args.port), StudioHandler)
    server.daemon_threads = True
    url = f"http://{args.host}:{args.port}"
    print(f"Team 7 Studio running at {url} (Compute: {STATE.device_info['name']})")
    if not args.no_browser and DIST_DIR.exists():
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        STATE.group.disconnect()
        server.server_close()


if __name__ == "__main__":
    main()
