"""No-retune, physically positioned, axis-specific occlusion diagnostic for D0."""

import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

from evaluation.layout_pose import benchmark as optical
from learning.d0_baselines import rollout, summarize, write_csv
from learning.datasets import (Frame, iter_trials, longest_uncovered_segment, np, project, visible)
from pose.fusion import AssociationLabels

TARGETS = ('front', 'side', 'top')
RADIUS = optical.HAND_RADIUS
GAP = .008
METHODS = ('all', 'cross_verify', 'recovery', 'gt_correspondence')


def iter_axis_trials(occluded=True):
    """Paired clear/sphere views share D0 poses and Phase III per-feature optical draws."""
    for item, target in product(iter_trials('lockbox', regimes=('clean',)), range(3)):
        direction = np.asarray(item.rig[target].center) / np.linalg.norm(item.rig[target].center)
        rows = []
        for index, frame in enumerate(item.frames):
            truth = frame.truth
            support = float(np.max((item.box.vertices @ truth.rotation.T) @ direction))
            sphere = truth.translation + (support + RADIUS + GAP) * direction if occluded else np.full(3, 10.)
            lost = np.zeros((4, 12), bool)
            labels, detections = [], []
            for camera_id, camera in enumerate(item.rig):
                geometry = optical.geometry(camera, truth, sphere)
                if occluded and camera_id == target:
                    lost[camera_id, geometry.edge_ids[~geometry.usable_edges]] = True
                sample = optical.sample_view(geometry, optical.noise_for(item.seed, index, camera_id), 'hand')
                labels.append(AssociationLabels(sample.corner_labels, sample.edge_labels))
                detections.append(sample.detections)
            fault = np.zeros(4, bool)
            fault[target] = occluded
            rows.append(Frame(tuple(detections), truth, tuple(labels), fault,
                              frame.corner_visible, frame.edge_visible, lost))
        condition = 'sphere' if occluded else 'clear'
        yield replace(item, id=f'{item.id}-{TARGETS[target]}-{condition}',
                      regime=f'{TARGETS[target]}_{condition}', frames=tuple(rows))


def d0_rectangle_counts():
    result = dict(active_camera_frames=0, geometric_lost=0, partially_clipped_spans=0, total_span_tests=0)
    for item in iter_trials('lockbox'):
        for index, frame in enumerate(item.frames):
            for fault in item.faults:
                if fault.kind != 'occlusion' or not fault.start <= index < fault.stop:
                    continue
                camera = item.rig[fault.camera]
                result['active_camera_frames'] += 1
                result['geometric_lost'] += int(frame.occlusion_lost[fault.camera].sum())
                midpoint = project(np.zeros((1, 3)), frame.truth, camera)[0][0]
                half = np.array([69, 52]) * fault.severity
                delta = np.asarray(fault.offset[:2]) * 24 + [18 * np.sin(index * .22), 15 * np.cos(index * .16)]
                low, high = midpoint + delta - half, midpoint + delta + half
                _, _, _, edges = visible(item.box, frame.truth, camera)
                for edge in edges:
                    piece = longest_uncovered_segment(edge, low, high)
                    length = np.linalg.norm(piece[1] - piece[0])
                    if 9 <= length < np.linalg.norm(edge[1] - edge[0]) - 1e-8:
                        result['partially_clipped_spans'] += 1
                    result['total_span_tests'] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    import cv2
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    rows, losses = [], dict.fromkeys(TARGETS, 0)
    for occluded in (False, True):
        for item in iter_axis_trials(occluded):
            target, condition = item.regime.split('_')
            truth_lost = sum(int(frame.occlusion_lost.sum()) for frame in item.frames)
            losses[target] += truth_lost
            for method in METHODS:
                rows.extend(dict(row, target_axis=target, occlusion_condition=condition) for row in rollout(item, method))
            print(f'{target} / {condition} / seed {item.seed}: {truth_lost} geometrically lost edges', flush=True)
    if min(losses.values()) <= 0:
        raise RuntimeError('Axis fixture did not produce a nonzero target-edge loss on every axis')
    args.output.mkdir(parents=True)
    write_csv(args.output / 'frames.csv', rows)
    write_csv(args.output / 'summary.csv', summarize(rows))
    counts = dict(d0_rectangles=d0_rectangle_counts(), sphere_losses_by_axis=losses, sphere_radius_m=RADIUS,
                  support_plane_gap_m=GAP, trajectories=2, frames_per_trajectory=12, methods=METHODS)
    (args.output / 'protocol.json').write_text(json.dumps(counts, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
