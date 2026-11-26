"""Time only four-view estimated hybrid frame fitting on fixed synthetic inputs."""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from experiments.fusion.fusion_poc import LAYOUTS, cameras, fit_frame, initial_pose, simulate_frame, trajectory


def detection_sequence(layout, seed, frames):
    rig = cameras(layout)
    return [
        tuple((camera, detections) for camera, (detections, _) in zip(
            rig, simulate_frame(trajectory(frame), rig, "stress", seed, frame), strict=True,
        ))
        for frame in range(frames)
    ]


def timings(sequences):
    elapsed = []
    for sequence in sequences:
        pose = initial_pose()
        for views in sequence:
            start = perf_counter_ns()
            pose, _, _ = fit_frame(pose, views, "hybrid")
            elapsed.append((perf_counter_ns() - start) / 1e6)
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results") / "latency.json")
    args = parser.parse_args()
    frames = 75
    seeds = (42, 43, 44)
    results = {}
    for layout in LAYOUTS:
        sequences = [detection_sequence(layout, seed, frames) for seed in seeds]
        timings(sequences[:1])  # Warm up with a full independent tracking pass.
        elapsed = timings(sequences)
        results[layout] = {
            "timed_frames": len(elapsed),
            "p50_ms": float(np.percentile(elapsed, 50)),
            "p95_ms": float(np.percentile(elapsed, 95)),
        }
    result = dict(
        platform=platform.platform(), processor=platform.processor(), python=platform.python_version(),
        numpy=np.__version__, scenario="stress", seeds=seeds, frames_per_trial=frames,
        warmup_frames_per_layout=frames, results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
