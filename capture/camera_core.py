"""Threaded acquisition and timestamped multi-camera recording.

OpenCV VideoCapture/VideoWriter perform device I/O and encoding. Device discovery
uses lukehugh/cv2_enumerate_cameras (MIT). Session orchestration is project code.
See CAMERA_GUIDE.txt for sources, timestamp semantics and limitations.
"""
import csv
import json
import math
import queue
import shutil
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2

KST = timezone(timedelta(hours=9))
OPEN_LOCK = threading.Lock()


def discover(backend=cv2.CAP_DSHOW):
    from cv2_enumerate_cameras import enumerate_cameras
    return [dict(index=c.index, name=c.name, path=c.path, backend=c.backend)
            for c in enumerate_cameras(backend)]


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class CameraWorker(threading.Thread):
    def __init__(self, slot, device, size, fps, publish):
        super().__init__(daemon=True, name=f'capture-{slot}')
        self.slot, self.device = slot, device
        self.size, self.requested_fps, self.publish = size, fps, publish
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.last_ns = 0
        self.times = deque(maxlen=60)
        self.status, self.error = '연결 중', ''
        self.actual = {}

    def snapshot(self):
        with self.lock:
            rate = ((len(self.times)-1)*1e9/(self.times[-1]-self.times[0])
                    if len(self.times) > 1 and self.times[-1] > self.times[0] else 0)
            return self.frame, self.sequence, self.last_ns, rate, self.status, self.error

    def run(self):
        cap = None
        try:
            # DirectShow device initialization is serialized; reads remain independent.
            with OPEN_LOCK:
                if self.stop_event.is_set():
                    return
                cap = cv2.VideoCapture(self.device['index'], self.device['backend'])
                if not cap.isOpened():
                    raise RuntimeError('열기 실패: 다른 앱의 카메라 사용 여부를 확인하세요.')
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
                self.actual = dict(backend=cap.getBackendName(),
                                   reported_fps=cap.get(cv2.CAP_PROP_FPS))
            failures = 0
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                received_ns = time.perf_counter_ns()
                wall_ns = time.time_ns()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError('영상 수신 실패: USB 연결 또는 다른 앱을 확인하세요.')
                    self.stop_event.wait(.1)
                    continue
                failures = 0
                self.actual['size_px'] = [frame.shape[1], frame.shape[0]]
                with self.lock:
                    self.frame, self.last_ns = frame, received_ns
                    self.sequence += 1
                    sequence = self.sequence
                    self.times.append(received_ns)
                    self.status = '연결됨'
                self.publish(self.slot, sequence, received_ns, wall_ns, frame)
        except Exception as exc:
            with self.lock:
                self.status, self.error = '오류', str(exc)
        finally:
            if cap is not None:
                cap.release()
            with self.lock:
                if not self.error:
                    self.status = '연결 해제'


class Recording:
    """Bounded, nonblocking acquisition queue; video and CSV share frame order."""
    def __init__(self, directory, cameras, fps, note='', queue_size=48):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(directory).free < 300*1024*1024:
            raise RuntimeError('저장 공간이 300 MB 미만입니다.')
        self.folder = directory / (datetime.now(KST).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
        self.folder.mkdir()
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event, self.done = threading.Event(), threading.Event()
        self.error = ''
        self.start_ns = self.stop_ns = None
        self.counts = {slot: 0 for slot in cameras}
        self.dropped = {slot: 0 for slot in cameras}
        self.last_written = {}
        self.resources = {}
        self.metadata = dict(status='preparing', timezone='Asia/Seoul',
                             created_at=datetime.now(KST).isoformat(), note=note,
                             codec='MJPG', container='AVI', requested_fps=fps,
                             playback_fps='Per camera, estimated from preview at recording start; see cameras.*.encoding_fps. CSV remains authoritative.',
                             timestamp_definition='Host timestamp immediately AFTER VideoCapture.read(); NOT sensor exposure time.',
                             synchronization='Shared software recording gate only; no hardware synchronization. Use CSV and a visible common signal for alignment.',
                             frames='Unmodified BGR frames; no resampling, duplication, or overlays; no audio.',
                             cameras=cameras, opencv_version=cv2.__version__)
        try:
            for slot, info in cameras.items():
                path = self.folder / f'camera_{slot}.avi'
                info.setdefault('encoding_fps', fps)
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'),
                                         info['encoding_fps'], tuple(info['actual']['size_px']))
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f'카메라 {slot} 영상 파일 생성 실패')
                try:
                    file = (self.folder / f'camera_{slot}_timestamps.csv').open('w', newline='', encoding='utf-8')
                except Exception:
                    writer.release()
                    raise
                csv_writer = csv.writer(file)
                self.resources[slot] = writer, file, csv_writer
                csv_writer.writerow(['video_frame_index', 'capture_sequence', 'host_monotonic_ns',
                                     'host_unix_ns', 'seconds_from_record_start'])
            write_json(self.folder / 'session.json', self.metadata)
        except Exception as exc:
            for writer, file, _ in self.resources.values():
                writer.release()
                file.close()
            self.metadata.update(status='preparation_failed', error=str(exc))
            write_json(self.folder / 'session.json', self.metadata)
            raise

    def begin(self):
        self.start_ns = time.perf_counter_ns()
        self.metadata.update(status='recording', start_monotonic_ns=self.start_ns,
                             start_unix_ns=time.time_ns())
        write_json(self.folder / 'session.json', self.metadata)
        self.thread = threading.Thread(target=self._write_loop, daemon=True, name='video-writer')
        self.thread.start()

    def push(self, slot, sequence, stamp, wall, frame):
        if self.start_ns is None or stamp < self.start_ns or self.stop_event.is_set() or self.error:
            return
        try:
            self.queue.put_nowait((slot, sequence, stamp, wall, frame))
        except queue.Full:
            self.dropped[slot] += 1

    def finish(self):
        self.stop_ns = time.perf_counter_ns()
        self.stop_event.set()

    def _write_loop(self):
        next_disk_check = 0
        try:
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    slot, sequence, stamp, wall, frame = self.queue.get(timeout=.1)
                except queue.Empty:
                    continue
                now = time.monotonic()
                if now >= next_disk_check:
                    if shutil.disk_usage(self.folder).free < 100*1024*1024:
                        raise RuntimeError('남은 디스크 공간이 100 MB 미만으로 녹화를 중단했습니다.')
                    next_disk_check = now + 1
                expected = self.metadata['cameras'][slot]['actual']['size_px']
                if [frame.shape[1], frame.shape[0]] != expected:
                    raise RuntimeError(f'카메라 {slot}의 해상도가 녹화 중 변경되었습니다.')
                if stamp <= self.last_written.get(slot, -1):
                    raise RuntimeError(f'카메라 {slot} 타임스탬프 순서 오류')
                writer, file, csv_writer = self.resources[slot]
                writer.write(frame)
                csv_writer.writerow([self.counts[slot], sequence, stamp, wall,
                                     f'{(stamp-self.start_ns)/1e9:.9f}'])
                self.counts[slot] += 1
                self.last_written[slot] = stamp
                if self.counts[slot] % 30 == 0:
                    file.flush()
        except Exception as exc:
            self.error = str(exc)
        finally:
            for writer, file, _ in self.resources.values():
                try:
                    writer.release()
                    file.close()
                except Exception as exc:
                    self.error = self.error or str(exc)
            self.metadata.update(status='error' if self.error else 'completed', error=self.error,
                                 stop_monotonic_ns=self.stop_ns or time.perf_counter_ns(),
                                 written_frames=self.counts, queue_dropped_frames=self.dropped,
                                 finished_at=datetime.now(KST).isoformat())
            try:
                write_json(self.folder / 'session.json', self.metadata)
            except Exception as exc:
                self.error = self.error or f'메타데이터 저장 실패: {exc}'
            self.done.set()


class CaptureGroup:
    def __init__(self):
        self.workers = {}
        self.lock = threading.Lock()
        self.recording = None

    def connect(self, selected, size=(1280, 720), fps=30):
        if self.workers:
            raise RuntimeError('기존 카메라 연결을 먼저 해제하세요.')
        ids = [(d['backend'], d['index']) for d in selected.values()]
        if not 1 <= len(ids) <= 4 or len(ids) != len(set(ids)):
            raise ValueError('서로 다른 카메라를 1~4대 선택하세요.')
        for slot, device in selected.items():
            self.workers[slot] = CameraWorker(slot, device, size, fps, self.publish)
        for worker in self.workers.values():
            worker.start()

    def publish(self, *args):
        with self.lock:
            if self.recording:
                self.recording.push(*args)

    def ready(self):
        now = time.perf_counter_ns()
        return bool(self.workers) and all(
            w.snapshot()[0] is not None and w.snapshot()[4] == '연결됨'
            and now-w.snapshot()[2] < 2_000_000_000 for w in self.workers.values())

    def start_recording(self, directory, fps, note=''):
        with self.lock:
            if self.recording:
                raise RuntimeError('이미 녹화 중입니다.')
            if not self.ready():
                raise RuntimeError('선택한 모든 카메라의 영상 수신을 먼저 확인하세요.')
            cameras = {slot: dict(device=w.device, actual=dict(w.actual),
                                  requested_size_px=list(w.size), requested_fps=w.requested_fps)
                       for slot, w in self.workers.items()}
            for slot, worker in self.workers.items():
                _, sequence, _, rate, _, _ = worker.snapshot()
                cameras[slot]['encoding_fps'] = (round(rate, 3) if sequence >= 10
                    and math.isfinite(rate) and .5 <= rate <= fps*1.2 else fps)
            rec = Recording(directory, cameras, fps, note)
            try:
                rec.begin()
            except Exception as exc:
                for writer, file, _ in rec.resources.values():
                    writer.release()
                    file.close()
                rec.metadata.update(status='start_failed', error=str(exc))
                try:
                    write_json(rec.folder / 'session.json', rec.metadata)
                except OSError:
                    pass
                raise
            self.recording = rec
            return rec

    def stop_recording(self):
        with self.lock:
            rec, self.recording = self.recording, None
            if rec:
                rec.finish()
            return rec

    def disconnect(self):
        rec = self.stop_recording()
        for worker in self.workers.values():
            worker.stop_event.set()
        return rec

    def clear_stopped(self):
        if any(w.is_alive() for w in self.workers.values()):
            return False
        self.workers.clear()
        return True
