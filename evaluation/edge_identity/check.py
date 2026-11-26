"""Post-fit edge-ID validation on the original Phase III synthetic hand trials."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

from evaluation.layout_pose import benchmark as source
from pose.fusion import EDGES, build_problem, project, solve_pose

np = source.np
LAYOUTS = {'A_same': (1, -1, 1), 'B_opposite': (-1, 1, -1),
           'upper_coverage': (-1, 1, 1), 'upper_conditioning': (-1, -1, 1)}
FRAMES = 28
NORMAL_LIMIT_PX = 2.5
ANGLE_LIMIT_DEG = 8.
OVERLAP_LIMIT = .60
VIEW_PAIRS = tuple(combinations(range(4), 2))


def gate_geometry(predicted, observed):
    """A line is constrained by normal RMS, unsigned angle, and finite interval overlap."""
    model_delta = predicted[:, 1] - predicted[:, 0]
    obs_delta = observed[:, 1] - observed[:, 0]
    lengths = np.linalg.norm(model_delta, axis=1)
    obs_lengths = np.linalg.norm(obs_delta, axis=1)
    nondegenerate = (lengths > 1e-8) & (obs_lengths > 1e-8)
    model_direction = model_delta / np.maximum(1e-8, lengths[:, None])
    obs_direction = obs_delta / np.maximum(1e-8, obs_lengths[:, None])
    obs_normal = obs_direction[:, ::-1] * [-1, 1]
    normal_distances = np.einsum('nei,ni->ne', predicted - observed[:, :1], obs_normal)
    distance_rms = np.sqrt(np.mean(normal_distances**2, axis=1))
    angle_degrees = np.degrees(np.arccos(np.clip(np.abs(np.sum(model_direction * obs_direction, axis=1)), 0, 1)))
    observed_endpoints = np.einsum('nei,ni->ne', observed - predicted[:, :1], model_direction)
    lower, upper = observed_endpoints.min(axis=1), observed_endpoints.max(axis=1)
    shared = np.maximum(0, np.minimum(lengths, upper) - np.maximum(0, lower))
    overlap = shared / np.maximum(1e-8, upper - lower)
    return nondegenerate & (distance_rms <= NORMAL_LIMIT_PX) & (angle_degrees <= ANGLE_LIMIT_DEG) & (overlap >= OVERLAP_LIMIT)


def keep_claims(fit, camera, detected, pairs):
    if not fit.valid or not len(pairs):
        return pairs[:0]
    uv, depths = project(source.BOX.vertices, fit.pose, camera)
    model_ids, observed_ids = pairs.T
    endpoints = EDGES[model_ids]
    keep = np.all(depths[endpoints] > 0, axis=1)
    keep &= gate_geometry(uv[endpoints], detected.segments[observed_ids])
    return pairs[keep]


def additional_pair_counts(samples, claims):
    detected = np.zeros((4, len(EDGES)), dtype=bool)
    assigned = np.full((4, len(EDGES)), -2, dtype=int)
    for index, (sample, rows) in enumerate(zip(samples, claims, strict=True)):
        labels = sample.edge_labels
        detected[index, labels[labels >= 0]] = True
        model, candidate = rows.T
        assigned[index, model] = labels[candidate]
    opportunities = false_clutter = false_physical = 0
    for first, second in VIEW_PAIRS:
        opportunities += np.count_nonzero(detected[first] & detected[second])
        both = (assigned[first] != -2) & (assigned[second] != -2)
        clutter = (assigned[first] == -1) | (assigned[second] == -1)
        wrong = (assigned[first] != np.arange(len(EDGES))) | (assigned[second] != np.arange(len(EDGES)))
        false_clutter += np.count_nonzero(both & clutter)
        false_physical += np.count_nonzero(both & ~clutter & wrong)
    return dict(detected_pair_opportunities=int(opportunities), false_pairs_with_clutter=int(false_clutter),
                false_pairs_physical_only=int(false_physical))


def original_rows():
    path = Path(source.__file__).with_name('results') / 'frame_metrics.csv'
    with path.open(encoding='utf-8', newline='') as handle:
        keep = {source.layout_name(signs) for signs in LAYOUTS.values()}
        return {(int(row['seed']), int(row['frame']), row['layout']): row for row in csv.DictReader(handle)
                if row['distance_mode'] == 'cube_vertex' and row['condition'] == 'hand' and row['method'] == 'hybrid'
                and row['layout'] in keep}


def compare_original(expected, mm, degrees, metrics):
    checks = ('multiview_pairs', 'multiview_pairs_correct', 'detected_edges_2plus', 'correct_edges_2plus',
              'designated_edges_lost', 'designated_edges_correct_elsewhere')
    if (abs(float(expected['translation_mm']) - mm) > 1e-8 or
            abs(float(expected['rotation_deg']) - degrees) > 1e-8 or
            any(int(expected[key]) != metrics[key] for key in checks)):
        raise RuntimeError('Ungated replay differs from the existing layout_pose hand result')


def evaluate():
    existing = original_rows()
    rows = []
    for name, signs in LAYOUTS.items():
        layout = source.layout_name(signs)
        rig = source.camera_rig('cube_vertex', signs)
        for seed in source.SEEDS:
            trajectory, prior = source.trajectory(seed, FRAMES)
            for frame, truth in enumerate(trajectory):
                hand, direction = source.hand_for(truth, seed, frame)
                geometry = tuple(source.geometry(camera, truth, hand) for camera in rig)
                samples = tuple(source.sample_view(data, source.noise_for(seed, frame, index), 'hand')
                                for index, data in enumerate(geometry))
                detections = tuple(sample.detections for sample in samples)
                full_fit, matches = source.fit_frame(prior, rig, detections, 'hybrid')
                pairs = tuple(match.segment_pairs for match in matches)
                leave_one_out = tuple(solve_pose(prior, build_problem(source.BOX, matches[:index] + matches[index + 1:]))
                                      for index in range(len(rig)))
                in_fit = tuple(keep_claims(full_fit, camera, view, claims)
                               for camera, view, claims in zip(rig, detections, pairs, strict=True))
                out_fit = tuple(keep_claims(fit, camera, view, claims)
                                for fit, camera, view, claims in zip(leave_one_out, rig, detections, pairs, strict=True))
                mm, degrees = source.errors(full_fit.pose, truth)
                if full_fit.valid:
                    prior = full_fit.pose
                designated = source.designated_camera('hand', direction, signs)
                for stage, claimed, usable in (('raw', pairs, len(rig)),
                                              ('in_fit_gate', in_fit, len(rig) if full_fit.valid else 0),
                                              ('left_camera_out_gate', out_fit, sum(fit.valid for fit in leave_one_out))):
                    edges = source.evaluate_edges(samples, tuple(SimpleNamespace(segment_pairs=rows) for rows in claimed),
                                                  geometry, designated, 'hand')
                    if stage == 'raw':
                        compare_original(existing[seed, frame, layout], mm, degrees, edges)
                    counts = additional_pair_counts(samples, claimed)
                    if edges['multiview_pairs'] != (edges['multiview_pairs_correct'] + counts['false_pairs_with_clutter'] +
                                                   counts['false_pairs_physical_only']):
                        raise RuntimeError('Claim classes do not partition all pair claims')
                    rows.append(dict(name=name, layout=layout, seed=seed, frame=frame, stage=stage,
                                     designated=source.VIEWS[designated] if designated is not None else '',
                                     fit_valid=full_fit.valid, validation_camera_fits_valid=int(usable),
                                     translation_mm=mm, rotation_deg=degrees, **edges, **counts))
    return rows


def summarize(rows):
    summary = []
    count_fields = ('multiview_pairs', 'multiview_pairs_correct', 'detected_pair_opportunities', 'detected_edges_2plus',
                    'correct_edges_2plus', 'false_pairs_with_clutter', 'false_pairs_physical_only',
                    'designated_edges_lost', 'designated_edges_correct_elsewhere', 'validation_camera_fits_valid')
    for name in LAYOUTS:
        for stage in ('raw', 'in_fit_gate', 'left_camera_out_gate'):
            subset = [row for row in rows if row['name'] == name and row['stage'] == stage]
            axis = [row for row in subset if row['designated'] in source.VIEWS[:3]]
            counts = {key: int(sum(row[key] for row in subset)) for key in count_fields}
            axis_lost = int(sum(row['designated_edges_lost'] for row in axis))
            axis_correct = int(sum(row['designated_edges_correct_elsewhere'] for row in axis))
            precision = source.percentage(counts['multiview_pairs_correct'], counts['multiview_pairs'])
            recall = source.percentage(counts['multiview_pairs_correct'], counts['detected_pair_opportunities'])
            edge_recall = source.percentage(counts['correct_edges_2plus'], counts['detected_edges_2plus'])
            summary.append(dict(name=name, layout=source.layout_name(LAYOUTS[name]), stage=stage, frames=len(subset),
                                correct_pair_precision_pct=precision, correct_pair_recall_pct=recall,
                                correct_physical_edge_recall_2plus_pct=edge_recall,
                                axis_rescue_correct=axis_correct, axis_rescue_lost=axis_lost,
                                axis_rescue_correct_pct=source.percentage(axis_correct, axis_lost), **counts))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    rows = evaluate()
    args.output.mkdir(parents=True)
    source.write_csv(args.output / 'validation_frames.csv', rows)
    source.write_csv(args.output / 'summary.csv', summarize(rows))
    config = dict(source='phase3.layout_pose.benchmark', condition='hand', distance_mode='cube_vertex',
                  tracker='hybrid, unchanged; neither validation route feeds back to tracking', seeds=source.SEEDS,
                  original_raw_frame_metrics_verified=len(rows) // 3, frames_per_seed=FRAMES, layouts=LAYOUTS,
                  gate=dict(normal_rms_px=NORMAL_LIMIT_PX, max_unsigned_angle_deg=ANGLE_LIMIT_DEG,
                            min_finite_overlap_of_observed_fraction=OVERLAP_LIMIT),
                  leave_camera_out='Same pre-fit causal prior; target camera excluded from this frame fit only')
    (args.output / 'protocol.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print(f'Verified {len(rows) // 3} raw replay frames; wrote {len(rows)} raw/post-fit rows')


if __name__ == '__main__':
    main()
