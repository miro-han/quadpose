"""Project adapter for unmodified official Meta SAM 2.1 video predictor."""
import gc
import json
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'third_party' / 'sam2'
CHECKPOINT = ROOT / 'models' / 'sam2.1_hiera_tiny.pt'
CONFIG = 'configs/sam2.1/sam2.1_hiera_t.yaml'


def save_png(path, image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('PNG encoding failed')
    data.tofile(str(path))


def visualize(frame, mask):
    overlay = frame.copy()
    overlay[mask] = (frame[mask]*.55 + np.array([40, 210, 60])*.45).astype(np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.2), 15, 45, L2gradient=True)
    # Include a narrow boundary band: the output remains an edge candidate map.
    area = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    selected = cv2.bitwise_and(edges, edges, mask=area)
    return overlay, selected


class SamSession:
    def __init__(self):
        self.predictor = None
        self.state = None
        self.stop = threading.Event()
        self.folder = None
        self.metadata = {}

    def ensure_model(self, notify):
        if self.predictor is not None:
            return
        if not (SOURCE/'sam2').is_dir() or not CHECKPOINT.exists():
            raise RuntimeError('SAM 2 소스 또는 모델이 없습니다. setup_sam2.py를 실행하세요.')
        sys.path.insert(0, str(SOURCE))
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        notify(f'SAM 2.1 Tiny 모델 준비 중 · {self.device.upper()}')
        self.predictor = build_sam2_video_predictor(CONFIG, str(CHECKPOINT),
                                                  device=self.device, apply_postprocessing=False)

    def prepare(self, video, start, limit, notify):
        self.stop.clear()
        self.state = None
        gc.collect()
        self.ensure_model(notify)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError('영상을 열 수 없습니다.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start < 0 or start >= total or limit < 1 or not np.isfinite(fps) or fps <= 0:
            cap.release()
            raise ValueError('시작 프레임과 처리 길이를 확인하세요.')
        count = min(limit, total-start)
        # Official video loader keeps normalized 1024x1024 RGB frames in memory.
        # Limit this CPU-first UI to bounded clips instead of exhausting laptop RAM.
        if count > 120:
            cap.release()
            raise ValueError('이 노트북용 첫 버전은 한 구간 최대 120프레임입니다. 시작 프레임을 바꿔 구간별로 처리하세요.')
        name = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        self.folder = ROOT/'outputs'/'segmentation'/name
        frames = self.folder/'input_frames'
        frames.mkdir(parents=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        actual = 0
        try:
            for i in range(count):
                if self.stop.is_set():
                    raise RuntimeError('준비를 취소했습니다.')
                ok, frame = cap.read()
                if not ok:
                    break
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError('입력 프레임 저장 실패')
                encoded.tofile(str(frames/f'{i:06d}.jpg'))
                actual += 1
            if not actual:
                raise RuntimeError('처리할 프레임이 없습니다.')
        finally:
            cap.release()
        self.metadata = dict(source=str(Path(video).resolve()), start_frame=start,
                             frame_count=actual, nominal_fps=fps, status='preparing',
                             model='SAM 2.1 Hiera Tiny', config=CONFIG, device=self.device,
                             source_commit=(ROOT/'third_party'/'sam2_commit.txt').read_text().strip(),
                             postprocessing_cuda_extension=False,
                             input_conversion='Official JPEG-folder loader; JPEG quality 95; original pixel coordinates',
                             notes='Masks are model predictions, not ground truth. Mask-selected edges can include occlusion boundaries. No 3D pose or cross-camera association.',
                             prompt_history=[])
        self.save_metadata()
        notify(f'{actual}프레임을 모델에 준비 중… CPU에서는 시간이 걸릴 수 있습니다.')
        import torch
        with torch.inference_mode():
            self.state = self.predictor.init_state(str(frames), offload_video_to_cpu=True,
                                                   offload_state_to_cpu=True)
        self.metadata['status'] = 'ready'
        self.save_metadata()

    def save_metadata(self):
        if self.folder:
            path = self.folder/'session.json'
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            temp.replace(path)

    def preview(self, box, points, labels):
        import torch
        started = time.perf_counter()
        with torch.inference_mode():
            self.predictor.reset_state(self.state)
            _, _, logits = self.predictor.add_new_points_or_box(
                self.state, frame_idx=0, obj_id=1,
                box=np.asarray(box, np.float32),
                points=np.asarray(points, np.float32) if points else None,
                labels=np.asarray(labels, np.int32) if points else None)
            mask = (logits[0, 0] > 0).cpu().numpy()
        self.metadata['prompt_history'].append(dict(box=box, points=points, labels=labels,
                                                    source_frame=self.metadata['start_frame'],
                                                    preview_seconds=time.perf_counter()-started))
        self.metadata['status'] = 'mask_preview'
        save_png(self.folder/'initial_mask.png', mask.astype(np.uint8)*255)
        self.save_metadata()
        return mask

    def track(self, notify, show):
        import torch
        self.stop.clear()
        self.metadata['status'] = 'tracking'
        self.save_metadata()
        for name in ['masks', 'selected_edges']:
            (self.folder/name).mkdir(exist_ok=True)
        writer = None
        completed = 0
        start_time = time.perf_counter()
        cap = cv2.VideoCapture(self.metadata['source'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.metadata['start_frame'])
        try:
            with (self.folder/'frames.jsonl').open('w', encoding='utf-8') as log, torch.inference_mode():
                for index, _, logits in self.predictor.propagate_in_video(self.state):
                    if self.stop.is_set():
                        break
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError('추적 중 원본 프레임을 읽지 못했습니다.')
                    mask = (logits[0, 0] > 0).cpu().numpy()
                    overlay, edges = visualize(frame, mask)
                    save_png(self.folder/'masks'/f'{index:06d}.png', mask.astype(np.uint8)*255)
                    save_png(self.folder/'selected_edges'/f'{index:06d}.png', edges)
                    if writer is None:
                        writer = cv2.VideoWriter(str(self.folder/'tracking_preview.mp4'),
                                                 cv2.VideoWriter_fourcc(*'mp4v'), self.metadata['nominal_fps'],
                                                 (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError('결과 영상 저장 실패')
                    writer.write(overlay)
                    log.write(json.dumps(dict(local_frame=index,
                        source_frame=self.metadata['start_frame']+index,
                        mask_area_px=int(mask.sum()), empty_mask=not bool(mask.any()),
                        elapsed_seconds=time.perf_counter()-start_time))+'\n')
                    log.flush()
                    completed += 1
                    show(frame, mask, index)
                    notify(f'추적 {completed}/{self.metadata["frame_count"]} · {time.perf_counter()-start_time:.1f}초')
            self.metadata['status'] = 'cancelled' if self.stop.is_set() else 'completed'
        except Exception as exc:
            self.metadata.update(status='error', error=str(exc))
            raise
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            self.metadata.update(written_frames=completed, processing_seconds=time.perf_counter()-start_time)
            self.save_metadata()
        return self.folder
