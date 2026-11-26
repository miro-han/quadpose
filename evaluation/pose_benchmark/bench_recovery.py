"""Small, reproducible B04/B06 diagnostics with no ground-truth tracker input."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np

from pose.fusion import EDGE_AXES, Detections, Pose, rotation_vector, visible
from evaluation.layout_pose.benchmark import BOX, camera_rig

from pose.hypotheses import POLICIES, evaluate_hypotheses, pose_distance, select_pose
from pose.recovery import MODES, RecoveryTracker


CAMERAS = camera_rig('equal_radius', (-1, -1, 1))
EMPTY = Detections(np.empty((0, 2)), np.empty((0, 2, 2)))


def motion(index, jump=False):
    if jump:
        return Pose(rotation_vector(np.array([.24, -.28, .42 + .02 * index])),
                    np.array([.125 + .002 * index, -.035, .033]))
    return Pose(rotation_vector(np.array([.06, -.08, .12 + .055 * index])),
                np.array([.0095 * index, -.004 * index, .012]))


def observe(pose, cameras=CAMERAS, *, keep=range(4), fault=None, parallel=False, noise=0., seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for index, camera in enumerate(cameras):
        if index not in keep:
            rows.append(EMPTY)
            continue
        _, points, ids, lines = visible(BOX, pose, camera)
        if parallel:
            axes = EDGE_AXES[ids]
            major = int(np.argmax(np.bincount(axes, minlength=3)))
            points, lines = points[:0], lines[axes == major]
        points = points + rng.normal(0, noise, points.shape)
        lines = lines + rng.normal(0, noise, lines.shape)
        if index == fault:
            shift = np.array([-22., 0.])
            points, lines = points + shift, lines + shift
        rows.append(Detections(points[rng.permutation(len(points))], lines[rng.permutation(len(lines))]))
    return tuple(rows)


def scenes(seed):
    repeated = tuple(replace(CAMERAS[3], name=f'parallel_{i}') for i in range(4))
    for index in range(20):
        truth = motion(index if index < 10 else index - 10, jump=index >= 10)
        rig, slots, parallel = CAMERAS, range(4), False
        condition = 'clean'
        if index in (3, 4):
            slots, condition = (), 'short_dropout'
        elif 6 <= index < 10:
            slots, condition = (), 'dropout_before_jump'
        elif index in (13, 14):
            slots, condition = (0,), 'single_view'
        elif index in (15, 16):
            slots, condition = (0, 1), 'two_views'
        elif index in (17, 18):
            rig, parallel = repeated, True
            condition = 'near_parallel_noisy' if index == 17 else 'parallel_exact'
        elif index == 10:
            condition = 'jump_reappearance'
        measured = observe(truth, rig, keep=slots, parallel=parallel,
                           noise=0. if condition == 'parallel_exact' else .05, seed=seed + 13 * index)
        yield index, condition, truth, rig, measured


def trial(seed, mode):
    trace = []
    seed_pose = motion(0)
    tracker = RecoveryTracker(BOX, seed_pose, mode=mode)
    for index, condition, truth, cameras, detections in scenes(seed):
        outcome = tracker.step(index * .075, cameras, detections)
        measured = outcome.status == 'measured'
        errors = pose_distance(outcome.pose, truth) if outcome.pose is not None else (None, None)
        trace.append({'mode': mode, 'frame': index, 'condition': condition, 'status': outcome.status,
                      'reason': outcome.reason, 'rank': outcome.diagnostics.get('rank'),
                      'support_count': int(outcome.support.sum()), 'reacquired': outcome.reacquired,
                      'pnp_proposals': outcome.diagnostics['pnp_proposals'],
                      'elapsed_ms': outcome.diagnostics['recovery_elapsed_ms'],
                      'translation_mm': None if errors[0] is None else 1000 * errors[0],
                      'rotation_deg': errors[1], 'correct_measurement': (measured and errors[0] <= .02 and errors[1] <= 8.)})
    return trace


def row_summary(row):
    observed = []
    for index, residual in enumerate(row.residuals_px):
        observed.append({'camera': CAMERAS[index].name, 'held_out': index == row.omitted,
                         'median_px': float(residual) if np.isfinite(residual) else None,
                         'features': int(row.feature_counts[index]), 'inliers': int(row.inlier_counts[index]),
                         'supported': bool(row.supported[index]),
                         'association_jaccard': (float(row.association_stability[index])
                                                 if np.isfinite(row.association_stability[index]) else None)})
    return {'omitted': row.omitted, 'fit_views': row.fit_views, 'fit_valid': row.fit.valid,
            'in_fit_rms_px': float(row.fit.rms) if np.isfinite(row.fit.rms) else None,
            'rank': row.rank, 'condition': row.condition if np.isfinite(row.condition) else None,
            'min_eigenvalue': row.min_eigenvalue, 'directional_information': row.directional_information.tolist(),
            'translation_delta_m': row.translation_delta_m, 'rotation_delta_deg': row.rotation_delta_deg,
            'camera_observations': observed}


def compare_matrix(seed):
    truth, prior = motion(1), motion(0)
    report = {}
    for case, fault in [('clean', None), ('faulty_front', 0)]:
        data = observe(truth, fault=fault, noise=.25, seed=seed)
        matrix = evaluate_hypotheses(prior, CAMERAS, data, BOX)
        policies = {}
        for policy in POLICIES:
            choice = select_pose(matrix, policy)
            errors = pose_distance(choice.pose, truth) if choice.pose is not None else (None, None)
            clean_diagnostics = {key: value if not isinstance(value, float) or np.isfinite(value) else None
                                 for key, value in choice.diagnostics.items()}
            policies[policy] = {'status': choice.status, 'support': choice.support.tolist(),
                                'translation_mm': None if errors[0] is None else errors[0] * 1000,
                                'rotation_deg': errors[1], 'diagnostics': clean_diagnostics}
        report[case] = {'rows': [row_summary(row) for row in (*matrix.triples, matrix.all)], 'policies': policies}
    return report


def summarize(rows):
    output = {}
    for mode in MODES:
        traces = [row for row in rows if row['mode'] == mode]
        measured = [row for row in traces if row['status'] == 'measured']
        reappearance = [row for row in traces if row['frame'] >= 10 and row['correct_measurement']]
        output[mode] = {'frames': len(traces), 'measured': len(measured),
                        'correct_measured': sum(row['correct_measurement'] for row in measured),
                        'false_valid': sum(not row['correct_measurement'] for row in measured),
                        'predicted': sum(row['status'] == 'predicted' for row in traces),
                        'lost': sum(row['status'] == 'lost' for row in traces),
                        'first_correct_after_jump': reappearance[0]['frame'] if reappearance else None,
                        'recovery_frames_after_jump': reappearance[0]['frame'] - 10 if reappearance else None,
                        'p50_ms': float(np.median([row['elapsed_ms'] for row in traces])),
                        'p95_ms': float(np.percentile([row['elapsed_ms'] for row in traces], 95))}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=119)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent / 'results' / 'recovery_diagnostic')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    args.output.mkdir(parents=True, exist_ok=True)
    tables = compare_matrix(args.seed)
    rows = [row for mode in MODES for row in trial(args.seed, mode)]
    with (args.output / 'recovery_frames.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'seed': args.seed, 'sample_period_s': .075,
              'success_threshold': {'translation_mm': 20, 'rotation_deg': 8},
              'matrix': tables, 'recovery': summarize(rows)}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Wrote {args.output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
