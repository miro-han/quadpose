"""D0: deterministic Layout A trajectories and truth-separated learning packets."""

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pose.contracts import CAMERA_FIELDS, HYPOTHESIS_FIELDS, OMITTED, inference_packet
from pose.fusion import BoxModel, Pose, analyze_frame, choose_result, project, rotation_vector, visible
from pose.synthetic_benchmark import (Fault, TrialSpec, fault_schedule, longest_uncovered_segment,
                                      motion_function, pose_error, rig_camera, simulate_view)

REGIMES = ('clean', 'wrong_association', 'unequal_quality', 'occlusion', 'missing', 'extrinsics', 'time_skew', 'simultaneous')
SPLITS = {'train': ('train', 72200, 3), 'dev': ('validation', 74100, 2), 'lockbox': ('test', 76400, 2)}
CENTERS = ((0., -.65, 0.), (.65, 0., 0.), (0., 0., .65), (.65, -.65, .65))
NAMES = ('front', 'side', 'top', 'diagonal')
K = np.array([[900., 0., 480.], [0., 900., 360.], [0., 0., 1.]])
BOX = BoxModel((.190, .120, .085))
FRAME_COUNT = 12
CADENCE_HZ = 30
POINT_CAP = 18
SEGMENT_CAP = 28
PACKET_FIELDS = ('camera_tokens', 'hypothesis_tokens', 'cross_view_residual_px', 'subset_membership_mask',
                 'residual_valid_mask', 'camera_has_matches', 'hypothesis_valid', 'view_support')
INPUT_FIELDS = (*PACKET_FIELDS, 'prior_rotation', 'prior_translation', 'box_size_m')
OBSERVATION_FIELDS = ('points', 'segments', 'point_mask', 'segment_mask', 'camera_center_m', 'camera_K',
                      'camera_Rcw', 'image_size_px', 'time_s', 'baseline_best_hypothesis', 'candidate_cost')


@dataclass(frozen=True)
class Frame:
    detections: tuple
    truth: object
    labels: tuple
    fault_mask: np.ndarray
    corner_visible: np.ndarray
    edge_visible: np.ndarray
    occlusion_lost: np.ndarray


@dataclass(frozen=True)
class Trial:
    id: str
    seed: int
    regime: str
    rig: tuple
    box: BoxModel
    initial: object
    frames: tuple[Frame, ...]
    motion: str
    faults: tuple[Fault, ...]


@dataclass(frozen=True)
class Split:
    inputs: dict
    targets: dict | None
    index: dict
    observations: dict


def make_rig():
    return tuple(rig_camera(name, np.asarray(mount), K.copy(), (960, 720))
                 for name, mount in zip(NAMES, CENTERS, strict=True))


def blocked_edges(camera, truth, edges, frame, fault):
    midpoint = project(np.zeros((1, 3)), truth, camera)[0][0]
    half_extent = np.array([69, 52]) * fault.severity
    shift = np.asarray(fault.offset[:2]) * 24 + [18 * np.sin(frame * .22), 15 * np.cos(frame * .16)]
    low, high = midpoint + shift - half_extent, midpoint + shift + half_extent
    visible_fragments = [longest_uncovered_segment(edge, low, high) for edge in edges]
    return np.array([np.linalg.norm(edge[1] - edge[0]) < 9 for edge in visible_fragments], dtype=bool)


def iter_trials(split, *, regimes=REGIMES, frames=FRAME_COUNT, repeats=None):
    """Yield frozen Layout A trajectories without running any pose estimator."""
    legacy_split, root, default_repeats = SPLITS[split]
    repeats = default_repeats if repeats is None else repeats
    regimes = tuple(regimes)
    if not 1 <= repeats <= 10 or frames < 8 or not regimes or len(set(regimes)) != len(regimes) or not set(regimes) <= set(REGIMES):
        raise ValueError('Need 1–10 repeats, at least 8 frames, and distinct known regimes')
    rig = make_rig()
    for repeat in range(repeats):
        for regime in regimes:
            seed = root + 97 * repeat + REGIMES.index(regime)
            motion_seed, prior_seed, fault_seed, noise_seed = np.random.SeedSequence(seed).spawn(4)
            family, pose_at = motion_function(legacy_split, np.random.default_rng(motion_seed), frames)
            pose_rng = np.random.default_rng(prior_seed)
            first = pose_at(0)
            initial = Pose(rotation_vector(pose_rng.normal(0, .040, 3)) @ first.rotation,
                           first.translation + pose_rng.normal(0, .0065, 3))
            spec = TrialSpec(legacy_split, regime, repeat, seed, frames)
            faults = fault_schedule(spec, np.random.default_rng(fault_seed))
            rendered = []
            for index in range(frames):
                active = {fault.camera: fault for fault in faults if fault.start <= index < fault.stop}
                detection_rows = []
                corner_visible, edge_visible = np.zeros((4, 8), bool), np.zeros((4, 12), bool)
                lost = np.zeros_like(edge_visible)
                truth = pose_at(index)
                for camera_id, camera in enumerate(rig):
                    rng = np.random.default_rng(np.random.SeedSequence([*noise_seed.generate_state(2), index, camera_id]))
                    fault = active.get(camera_id)
                    detection_rows.append(simulate_view(BOX, camera, pose_at, index, fault,
                                                        rng.uniform(.57, 1.62), rng))
                    corner_ids, _, edge_ids, edges = visible(BOX, truth, camera)
                    corner_visible[camera_id, corner_ids] = True
                    edge_visible[camera_id, edge_ids] = True
                    if fault is not None and fault.kind == 'occlusion':
                        lost[camera_id, edge_ids] = blocked_edges(camera, truth, edges, index, fault)
                detections, labels = zip(*detection_rows, strict=True)
                rendered.append(Frame(detections, truth, labels, np.isin(np.arange(4), list(active)),
                                      corner_visible, edge_visible, lost))
            yield Trial(f'D0-{split}-{seed}', seed, regime, rig, BOX, initial, tuple(rendered), family, faults)


def raw_inputs(item, frame, number):
    points = np.zeros((4, POINT_CAP, 2), np.float32)
    segments = np.zeros((4, SEGMENT_CAP, 2, 2), np.float32)
    point_mask, segment_mask = np.zeros(points.shape[:2], bool), np.zeros(segments.shape[:2], bool)
    for i, view in enumerate(frame.detections):
        count_points, count_segments = len(view.points), len(view.segments)
        if count_points > POINT_CAP or count_segments > SEGMENT_CAP:
            raise RuntimeError('Input exceeds the archived D0 schema padding limits')
        points[i, :count_points], point_mask[i, :count_points] = view.points, True
        segments[i, :count_segments], segment_mask[i, :count_segments] = view.segments, True
    return dict(points=points, segments=segments, point_mask=point_mask, segment_mask=segment_mask,
                camera_center_m=np.stack([camera.center for camera in item.rig]),
                camera_K=np.stack([camera.intrinsic for camera in item.rig]),
                camera_Rcw=np.stack([camera.rotation for camera in item.rig]),
                image_size_px=np.asarray([camera.image_size for camera in item.rig]), time_s=number / CADENCE_HZ)


def label_inputs(frame, available, ordered):
    corner_ids, edge_ids = np.full((4, POINT_CAP), -2), np.full((4, SEGMENT_CAP), -2)
    for i, labels in enumerate(frame.labels):
        corner_ids[i, :len(labels.corner_ids)] = labels.corner_ids
        edge_ids[i, :len(labels.edge_ids)] = labels.edge_ids
    errors = np.zeros((4, 2), np.float32)
    for i in np.flatnonzero(available):
        errors[i] = pose_error(ordered[i].pose, frame.truth)
    costs = np.where(available, np.sum((errors / [20, 8])**2, axis=1), np.inf)
    return dict(target_rotation=frame.truth.rotation, target_translation=frame.truth.translation,
                target_corner_ids=corner_ids, target_edge_ids=edge_ids,
                target_corner_visible=frame.corner_visible, target_edge_visible=frame.edge_visible,
                target_occlusion_lost=frame.occlusion_lost,
                target_fault_mask=frame.fault_mask, target_reliable_mask=~frame.fault_mask,
                target_reliability_known=np.ones(4, bool), target_valid=int(np.count_nonzero(~frame.fault_mask)) >= 3,
                target_hypothesis_error=errors, target_hypothesis_known=available,
                target_best_hypothesis=int(np.argmin(costs)) if available.any() else -1)


def export_split(directory, split, *, repeats=None, frames=FRAME_COUNT, regimes=REGIMES):
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {'inputs': [], 'observations': [], 'targets': [], 'index': []}
    trials, target_trials = [], []
    for item in iter_trials(split, repeats=repeats, frames=frames, regimes=regimes):
        prior = item.initial
        trials.append({'trial_id': item.id, 'seed': item.seed, 'motion': item.motion})
        target_trials.append({'trial_id': item.id, 'regime': item.regime, 'faults': [vars(fault) for fault in item.faults]})
        for number, frame in enumerate(item.frames):
            evidence = analyze_frame(prior, item.rig, frame.detections, item.box)
            packet = inference_packet(evidence)
            arrays['inputs'].append(dict(packet, prior_rotation=prior.rotation, prior_translation=prior.translation,
                                         box_size_m=item.box.size_m))
            ordered = tuple(evidence.hypotheses[i] for i in OMITTED)
            available = packet['hypothesis_valid']
            candidates = [i for i in range(4) if available[i]]
            best = min(candidates, key=lambda i: (ordered[i].rms, ordered[i].condition, int(OMITTED[i]))) if candidates else -1
            arrays['observations'].append(dict(raw_inputs(item, frame, number), baseline_best_hypothesis=best,
                                               candidate_cost=np.array([candidate.rms if valid else 0.
                                                                        for valid, candidate in zip(available, ordered, strict=True)])))
            arrays['targets'].append(label_inputs(frame, available, ordered))
            arrays['index'].append({'frame_key': f'{item.id}:{number:03}', 'trial_key': item.id,
                                    'frame': number, 'split': split})
            predicted = choose_result(evidence, 'reliable')
            if predicted.valid:
                prior = predicted.pose
    keys = np.array([row['frame_key'] for row in arrays['index']])
    for name, rows in arrays.items():
        data = {field: np.asarray([row[field] for row in rows]) for field in rows[0]}
        np.savez_compressed(directory / f'{name}.npz', **{**data, 'frame_key': keys})
    (directory / 'target_provenance.json').write_text(json.dumps(target_trials, indent=2) + '\n', encoding='utf-8')
    hashes = {archive.name: hashlib.sha256(archive.read_bytes()).hexdigest() for archive in sorted(directory.iterdir())}
    manifest = {'schema_version': 1, 'dataset': 'D0', 'split': split, 'frames': len(keys), 'trials': trials,
                'rig_centers_m': CENTERS, 'camera_slot_order': NAMES, 'K': K.tolist(), 'box_size_m': BOX.size_m,
                'image_size_px': [960, 720], 'camera_token_fields': CAMERA_FIELDS, 'hypothesis_token_fields': HYPOTHESIS_FIELDS,
                'candidate_order': ['T123', 'T124', 'T134', 'T234'], 'files_sha256': hashes}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def _read(path, allowed=None):
    with np.load(path, allow_pickle=False) as archive:
        fields = list(archive.files) if allowed is None else list(allowed)
        if allowed is not None and set(archive.files) != set(allowed) | {'frame_key'}:
            raise ValueError(f'Unexpected feature fields in {path.name}')
        return {name: archive[name] for name in fields}, archive['frame_key']


def load_split(root, split, *, with_targets=False):
    """Load never-oracle observations; targets require a separate explicit request."""
    if split not in SPLITS:
        raise ValueError(f'Unsupported split {split}')
    directory = Path(root) / split
    inputs, key = _read(directory / 'inputs.npz', INPUT_FIELDS)
    observations, observed_key = _read(directory / 'observations.npz', OBSERVATION_FIELDS)
    index, indexed_key = _read(directory / 'index.npz')
    if not np.array_equal(key, observed_key) or not np.array_equal(key, indexed_key) or len(set(key)) != len(key):
        raise ValueError('Input, observation, and index frame keys differ or repeat')
    targets = None
    if with_targets:
        targets, target_key = _read(directory / 'targets.npz')
        if not np.array_equal(key, target_key):
            raise ValueError('Targets do not join one-to-one with inputs')
        if not set(targets) - {'frame_key'} or any(not name.startswith('target_') for name in set(targets) - {'frame_key'}):
            raise ValueError('Target archive contains fields outside the target namespace')
    for namespace in (inputs, observations, index, targets or {}):
        if any(value.ndim < 1 or len(value) != len(key) for value in namespace.values()):
            raise ValueError('Every archived array must align on its first dimension with the frame index')
    if not (observations['point_mask'].dtype == observations['segment_mask'].dtype == bool):
        raise ValueError('Padding masks must be boolean')
    return Split(inputs, targets, index, observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=tuple(SPLITS), action='append')
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--frames', type=int, default=FRAME_COUNT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new dataset output directory')
    for split in args.split or SPLITS:
        result = export_split(args.output / split, split, repeats=args.repeats, frames=args.frames)
        print(f'{split}: {result["frames"]} frames', flush=True)


if __name__ == '__main__':
    main()
