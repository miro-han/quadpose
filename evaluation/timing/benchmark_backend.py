"""Time the two CPU fusion backends on identical precomputed held-out detections."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from time import perf_counter_ns

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

from pose.fusion import (
    analyze_frame, associate_view, build_problem, choose_result, solve_pose, view_match,
)
from pose.synthetic_benchmark import generate_trial, trial_specs
from pose.contracts import inference_packet


def all_joint(prior, cameras, detections, box):
    matches = tuple(view_match(camera, detections_view, *associate_view(prior, camera, detections_view, box))
                    for camera, detections_view in zip(cameras, detections, strict=True))
    return solve_pose(prior, build_problem(box, matches))


def diagnostics(prior, cameras, detections, box):
    evidence = analyze_frame(prior, cameras, detections, box)
    packet = inference_packet(evidence)
    return choose_result(evidence, 'reliable'), packet


def prepare(frames):
    workload, manifests = [], []
    for spec in trial_specs('test', trials=1, frames=frames):
        trial = generate_trial(spec)
        manifests.append({'id': spec.trial_id, 'seed': spec.seed, 'frames': spec.frames})
        prior = trial.initial
        for frame in trial.frames:
            arguments = (prior, trial.cameras, frame.detections, trial.box)
            # Freeze the reliable rolling prior for both measured branches. Ground truth is unused.
            evidence = analyze_frame(*arguments)
            fit = all_joint(*arguments)
            expected = evidence.all_fit
            same_rotation = np.allclose(fit.pose.rotation, expected.pose.rotation)
            same_translation = np.allclose(fit.pose.translation, expected.pose.translation)
            if fit.valid != expected.valid or not (same_rotation and same_translation):
                raise RuntimeError('All-joint path differs from the full evidence all-joint fit')
            workload.append(arguments)
            estimate = choose_result(evidence, 'reliable')
            if estimate.valid:
                prior = estimate.pose
    return workload, manifests


def measure(workload, repeats, warmup):
    methods = (('all_joint_only', all_joint), ('diagnostics_and_reliable', diagnostics))
    for index in range(warmup):
        for _, call in methods:
            call(*workload[index % len(workload)])
    timings = {name: [] for name, _ in methods}
    per_repeat = []
    for repeat in range(repeats):
        samples = {name: [] for name, _ in methods}
        for index, arguments in enumerate(workload):
            order = methods if (index + repeat) % 2 == 0 else reversed(methods)
            for name, call in order:
                start = perf_counter_ns()
                result = call(*arguments)
                elapsed_ms = (perf_counter_ns() - start) / 1_000_000
                samples[name].append(elapsed_ms)
                del result
        for name, values in samples.items():
            timings[name].extend(values)
        per_repeat.append({name: percentiles(values) for name, values in samples.items()})
    return timings, per_repeat


def percentiles(values):
    p50, p95 = np.quantile(values, [.50, .95])
    return {'samples': len(values), 'p50_ms': float(p50), 'p95_ms': float(p95)}


def cpu_name():
    if platform.system() == 'Windows':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return platform.processor()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=24, help='Frames per held-out trial')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=16, help='Untimed workload frames per method')
    parser.add_argument('--load-note', default='Shared desktop environment; background and concurrent loads uncontrolled.')
    args = parser.parse_args()
    if args.frames < 12 or args.repeats < 2 or args.warmup < 1 or args.output.exists():
        parser.error('Use >=12 frames, >=2 repeats, >=1 warmup frame and a new output path')
    workload, manifest = prepare(args.frames)
    timings, per_repeat = measure(workload, args.repeats, args.warmup)
    point_counts = [sum(len(view.points) for view in views) for _, _, views, _ in workload]
    segment_counts = [sum(len(view.segments) for view in views) for _, _, views, _ in workload]
    metrics = {name: percentiles(values) for name, values in timings.items()}
    report = {
        'measured_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(), 'cpu': cpu_name(), 'logical_cpus': os.cpu_count(),
        'python': platform.python_version(), 'numpy': np.__version__, 'blas_thread_limit_env': 1,
        'load_note': args.load_note, 'workload': 'phase2.fusion synthetic benchmark, held-out test split',
        'timed_input': 'precomputed unlabeled corner and segment detections, calibration, cuboid and shared prior',
        'excluded': ['simulation and rolling-prior preparation', 'camera I/O', 'video decoding',
                     'SAM / LK / image feature detection', 'serialization and disk I/O'],
        'families': manifest, 'unique_frames': len(workload), 'warmup_frames_per_method': args.warmup,
        'repeats': args.repeats, 'order': 'Alternates method-first order each frame and repeat',
        'unique_frame_points_sum_range': [min(point_counts), max(point_counts)],
        'unique_frame_segments_sum_range': [min(segment_counts), max(segment_counts)],
        'metrics': metrics, 'per_repeat': per_repeat,
        'p50_ratio': metrics['diagnostics_and_reliable']['p50_ms'] / metrics['all_joint_only']['p50_ms'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
