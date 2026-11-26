"""Layout A D0 lockbox benchmark for frozen classical methods and four controls."""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter_ns

from evaluation.pose_benchmark.benchmark import EMPTY, feature_counts, process_peak_rss
from learning.datasets import CADENCE_HZ, FRAME_COUNT, REGIMES, iter_trials, np
from pose.fusion import (analyze_frame, associate_view, build_problem, choose_result, oracle_pairs,
                         solve_pose, view_match)
from pose.hypotheses import evaluate_hypotheses, select_pose
from pose.synthetic_benchmark import pose_error

CLASSICAL = ('all', 'best', 'cross', 'cross_verify', 'cross_alternating', 'recovery', 'adaptive_nonlearning')
ORACLES = ('gt_correspondence', 'gt_hypothesis_selection', 'gt_measurement_weighting', 'gt_initialization')
METHODS = CLASSICAL + ORACLES
LIMITS = (20., 8.)


@dataclass(frozen=True)
class Outcome:
    pose: object
    measured: bool
    status: str
    edges: tuple
    support: tuple


def emitted(pose, valid, matches, support=None, status=None):
    included = tuple(bool(match.corners + match.edges) for match in matches) if support is None else tuple(support)
    pairs = tuple(match.segment_pairs if keep else match.segment_pairs[:0] for match, keep in zip(matches, included, strict=True))
    return Outcome(pose, valid, status or ('measured' if valid else 'lost'), pairs if valid else EMPTY,
                   included if valid else (False,) * 4)


def joint(prior, rig, data, box, labels=None, faults=None):
    matches = []
    for i, (camera, detected) in enumerate(zip(rig, data, strict=True)):
        if faults is not None and faults[i]:
            pairs = (EMPTY[i], EMPTY[i])
        elif labels is not None:
            pairs = oracle_pairs(labels[i])
        else:
            pairs = associate_view(prior, camera, detected, box)
        matches.append(view_match(camera, detected, *pairs))
    fit = solve_pose(prior, build_problem(box, matches))
    return emitted(fit.pose if fit.valid else prior, fit.valid, matches)


def apply_classical(method, prior, rig, data, box, tracker, timestamp):
    if method in ('all', 'gt_initialization'):
        return joint(prior, rig, data, box)
    if method == 'best':
        evidence = analyze_frame(prior, rig, data, box)
        chosen = choose_result(evidence, 'best')
        return emitted(chosen.pose, chosen.valid, evidence.matches, chosen.support)
    if tracker is None:
        choice = select_pose(evaluate_hypotheses(prior, rig, data, box), policy=method)
    else:
        choice = tracker.step(timestamp, rig, data)
    return Outcome(choice.pose if choice.pose is not None else prior, choice.valid, choice.status,
                   choice.segment_pairs if choice.valid else EMPTY, tuple(choice.support))


def apply_oracle(method, prior, item, frame):
    if method == 'gt_correspondence':
        return joint(prior, item.rig, frame.detections, item.box, labels=frame.labels)
    if method == 'gt_measurement_weighting':
        return joint(prior, item.rig, frame.detections, item.box, faults=frame.fault_mask)
    evidence = analyze_frame(prior, item.rig, frame.detections, item.box)
    candidates = [(index, fit) for index, fit in enumerate(evidence.hypotheses) if fit.valid]
    if not candidates:
        return emitted(prior, False, evidence.matches)

    def cost(candidate):
        index, fit = candidate
        millimetres, degrees = pose_error(fit.pose, frame.truth)
        return ((millimetres / LIMITS[0])**2 + (degrees / LIMITS[1])**2, index)

    omitted, fit = min(candidates, key=cost)
    return emitted(fit.pose, True, evidence.matches, [i != omitted for i in range(4)])


def make_tracker(method, item):
    if method not in ('recovery', 'adaptive_nonlearning'):
        return None
    from pose.recovery import RecoveryTracker
    return RecoveryTracker(item.box, item.initial, policy='cross', mode='full' if method == 'recovery' else 'adaptive')


def rollout(item, method):
    tracker = make_tracker(method, item)
    prior = item.frames[0].truth if method == 'gt_initialization' else item.initial
    previous_ids = {}
    rows = []
    for index, frame in enumerate(item.frames):
        start = perf_counter_ns()
        if method in ORACLES and method != 'gt_initialization':
            result = apply_oracle(method, prior, item, frame)
        else:
            result = apply_classical(method, prior, item.rig, frame.detections, item.box, tracker, index / CADENCE_HZ)
        latency = (perf_counter_ns() - start) / 1e6
        if result.measured:
            prior = result.pose
        mm, deg = pose_error(result.pose, frame.truth)
        accurate = mm <= LIMITS[0] and deg <= LIMITS[1]
        samples = tuple(type('ScoreLabels', (), {'edge_labels': view.edge_ids}) for view in frame.labels)
        counts, previous_ids = feature_counts(samples, result.edges, frame.occlusion_lost, previous_ids)
        rows.append(dict(trial=item.id, seed=item.seed, regime=item.regime, frame=index, method=method, oracle=method in ORACLES,
                         fault_active=bool(frame.fault_mask.any()), measured=result.measured, status=result.status,
                         successful=result.measured and accurate, false_valid=result.measured and not accurate,
                         translation_mm=mm, rotation_deg=deg, latency_ms=latency,
                         supported_views=sum(result.support), **counts))
    return rows


def percentage(count, total):
    return 100 * count / total if total else None


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for regime in ('pooled', row['regime']):
            groups[row['method'], regime, 'all'].append(row)
            if row['fault_active']:
                groups[row['method'], regime, 'fault_active'].append(row)
    summaries = []
    for (method, regime, scope), entries in groups.items():
        counts = {key: int(sum(row[key] for row in entries)) for key in
                  ('measured', 'successful', 'false_valid', 'edge_assigned', 'edge_correct', 'edge_detected',
                   'pair_claims', 'pair_correct', 'pair_available', 'rescue_lost', 'rescue_correct')}
        summary = dict(method=method, oracle=method in ORACLES, regime=regime, scope=scope, frames=len(entries), **counts,
                       success_pct=percentage(counts['successful'], len(entries)),
                       false_valid_pct=percentage(counts['false_valid'], counts['measured']),
                       edge_precision_pct=percentage(counts['edge_correct'], counts['edge_assigned']),
                       edge_recall_pct=percentage(counts['edge_correct'], counts['edge_detected']),
                       edge_f1_pct=percentage(2 * counts['edge_correct'], counts['edge_assigned'] + counts['edge_detected']),
                       pair_accuracy_pct=percentage(counts['pair_correct'], counts['pair_claims']),
                       pair_recall_pct=percentage(counts['pair_correct'], counts['pair_available']),
                       rescue_pct=percentage(counts['rescue_correct'], counts['rescue_lost']))
        for key in ('translation_mm', 'rotation_deg', 'latency_ms'):
            sample = np.array([row[key] for row in entries])
            summary[f'{key}_p50'], summary[f'{key}_p95'] = np.quantile(sample, [.5, .95])
            if key != 'latency_ms':
                summary[f'{key}_rmse'] = float(np.sqrt(np.mean(sample**2)))
        summaries.append(summary)
    return summaries


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'lockbox'), default='lockbox')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new result directory')
    import cv2
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)
    if cv2.ocl.useOpenCL() or cv2.getNumThreads() != 1:
        raise RuntimeError('Require CPU only and one OpenCV thread')
    rows = []
    regimes = REGIMES[:1] if args.smoke else REGIMES
    trials = []
    for item in iter_trials(args.split, regimes=regimes, repeats=1 if args.smoke else None):
        trials.append(dict(id=item.id, seed=item.seed, regime=item.regime, motion=item.motion))
        for method in METHODS:
            rows.extend(rollout(item, method))
        print(f'{item.id}: {item.regime}', flush=True)
    args.output.mkdir(parents=True)
    write_csv(args.output / 'frames.csv', rows)
    write_csv(args.output / 'summary.csv', summarize(rows))
    manifest = dict(dataset='D0', split=args.split, smoke=args.smoke, trials=trials, methods=CLASSICAL, oracles=ORACLES,
                    initial='one perturbed first-pose initialization; exact first frame only for gt_initialization',
                    success_limits_mm_deg=LIMITS, frames_per_trial=FRAME_COUNT, tool='CPU only',
                    subprocess_peak_working_set_mib=process_peak_rss(), prior='method-local causal rollout',
                    gt_hypothesis_cost='(translation_mm/20)^2 + (rotation_deg/8)^2',
                    gt_weighting='binary healthy/fault mask; does not change feature association')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
