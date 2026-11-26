"""Repeat the shipped 30-frame tracking session with an explicit device."""

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / 'data' / 'sample_videos' / 'WIN_20261124_20_05_34_Pro.mp4'
BASELINE = ROOT / 'outputs' / 'segmentation' / '20261124_215925_8f211c' / 'session.json'
BOX = [831.5766738660907, 272.9157667386609, 1237.451403887689, 706.7818574514039]
FRAME_COUNT = 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    import cv2
    import torch
    import torchvision

    from perception.segmentation.sam_session import CHECKPOINT, SamSession
    from perception.segmentation.setup_sam2 import digest

    cuda = args.device == 'cuda'
    if cuda and not torch.cuda.is_available():
        parser.error(f'CUDA unavailable in torch {torch.__version__} (runtime {torch.version.cuda})')
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    phases = {}

    def measure(name, operation):
        started = perf_counter()
        result = operation()
        if cuda:
            torch.cuda.synchronize()
        phases[name] = perf_counter() - started
        print(f'{name}: {phases[name]:.3f} s', flush=True)
        return result

    def ignore_gui_callback(*_):
        pass

    run = SamSession()
    measure('model_load', lambda: run.ensure_model(ignore_gui_callback))
    if run.device != args.device:
        raise RuntimeError(f'Requested {args.device}, but the tracking adapter selected {run.device}')
    model_parameter = next(run.predictor.parameters())
    if model_parameter.device.type != args.device:
        raise RuntimeError(f'Model parameters are on {model_parameter.device}, expected {args.device}')
    measure('prepare', lambda: run.prepare(VIDEO, 0, FRAME_COUNT, ignore_gui_callback))
    measure('prompt', lambda: run.preview(BOX, [], []))
    measure('track', lambda: run.track(ignore_gui_callback, ignore_gui_callback))
    if run.metadata['status'] != 'completed' or run.metadata['written_frames'] != FRAME_COUNT:
        raise RuntimeError('Tracking did not produce the expected 30 frames')

    rows = [json.loads(line) for line in
            (run.folder / 'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != FRAME_COUNT or [row['source_frame'] for row in rows] != list(range(FRAME_COUNT)):
        raise RuntimeError('Tracking log has missing, duplicate, or out-of-order frames')

    processing = run.metadata['processing_seconds']
    report = {
        'device': run.device,
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'torch_cuda_runtime': torch.version.cuda,
        'opencv': cv2.__version__,
        'torch_cpu_threads': torch.get_num_threads(),
        'gpu': torch.cuda.get_device_name() if cuda else None,
        'model_parameter_device': str(model_parameter.device),
        'precision': str(model_parameter.dtype),
        'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
        'gpu_peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20 if cuda else None,
        'video': VIDEO.relative_to(ROOT).as_posix(),
        'video_sha256': digest(VIDEO),
        'model': run.metadata['model'],
        'model_config': run.metadata['config'],
        'sam2_source_commit': run.metadata['source_commit'],
        'checkpoint_sha256': digest(CHECKPOINT),
        'start_frame': 0,
        'frame_count': FRAME_COUNT,
        'box': BOX,
        'empty_masks': sum(row['empty_mask'] for row in rows),
        'elapsed_seconds': phases,
        'total_seconds': sum(phases.values()),
        'tracking_seconds': processing,
        'tracking_fps': len(rows) / processing,
        'session': run.folder.relative_to(ROOT).as_posix(),
    }
    if BASELINE.is_file():
        baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
        report['historical_cpu_tracking_seconds'] = baseline['processing_seconds']
        report['historical_cpu_tracking_speedup'] = baseline['processing_seconds'] / processing
    path = run.folder / 'benchmark.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Benchmark result: {path}')


if __name__ == '__main__':
    main()
