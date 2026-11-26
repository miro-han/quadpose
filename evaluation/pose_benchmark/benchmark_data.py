"""Frozen synthetic inputs. Truth exists only here and in the scoring functions."""

from dataclasses import dataclass, replace
from pathlib import Path
import json

from evaluation.layout_pose import benchmark as source
from pose.fusion import EDGES, Detections, Pose, project, rotation_vector

np = source.np
CONFIG_PATH = Path(__file__).with_name('benchmark_config.json')
FAR = np.ones(3) * 10


@dataclass
class Frame:
    truth: Pose
    samples: tuple
    lost_by_occlusion: np.ndarray
    active: bool

    @property
    def detections(self):
        return tuple(sample.detections for sample in self.samples)


def config():
    return json.loads(CONFIG_PATH.read_text(encoding='utf-8'))


def motion(seed, regime, settings):
    poses, initial = source.trajectory(seed, settings['frames'])
    start, stop = settings['fault_start'], settings['fault_end_exclusive']
    for index, pose in enumerate(poses):
        if regime == 'self_occlusion':
            spin = .77 * np.sin(.24 * index)
            rotation = rotation_vector(np.array([.26, -.62, .58]) * spin) @ pose.rotation
        elif regime in ('rapid_rotation', 'reacquisition', 'combined'):
            speed = .55 if regime == 'rapid_rotation' else .16
            turn = speed * max(0, min(index - start + 1, stop - start))
            rotation = rotation_vector(np.array([.24, -.56, .79]) * turn) @ pose.rotation
        else:
            rotation = pose.rotation
        poses[index] = Pose(rotation, pose.translation)
    return poses, initial


def sample_change(sample, *, points=True, segments=None, labels=None):
    detections = sample.detections
    return source.Sample(Detections(detections.points if points else detections.points[:0],
                                    detections.segments if segments is None else segments),
                         sample.corner_labels if points else sample.corner_labels[:0],
                         sample.edge_labels if labels is None else labels)


def fragmentation(sample, rng):
    visible = sample.detections.segments
    if not len(visible):
        return sample
    split = rng.uniform(.39, .53, len(visible))
    direction = visible[:, 1] - visible[:, 0]
    left = np.stack((visible[:, 0], visible[:, 0] + (split - .04)[:, None] * direction), axis=1)
    right = np.stack((visible[:, 0] + (split + .05)[:, None] * direction, visible[:, 1]), axis=1)
    fragments = np.concatenate((left, right)) + rng.normal(0, .28, (2 * len(visible), 2, 2))
    ids = np.tile(sample.edge_labels, 2)
    order = rng.permutation(len(fragments))
    return sample_change(sample, segments=fragments[order], labels=ids[order])


def decoy_edges(sample, camera, pose, rng):
    # A coherent wrong silhouette yields unlabeled competing segments with no physical identity.
    offset = np.array([.013, -.017, .006])
    wrong_pose = Pose(rotation_vector(np.array([.08, -.13, .10])) @ pose.rotation, pose.translation + offset)
    _, _, _, wrong = source.visible(source.BOX, wrong_pose, camera)
    keep = rng.random(len(sample.edge_labels)) < .44
    mixed = np.concatenate((sample.detections.segments[keep], wrong + rng.normal(0, 1, wrong.shape)))
    labels = np.concatenate((sample.edge_labels[keep], np.full(len(wrong), -1)))
    order = rng.permutation(len(labels))
    return sample_change(sample, segments=mixed[order], labels=labels[order])


def trial(seed, regime, signs, settings):
    poses, initial = motion(seed, regime, settings)
    rig = source.camera_rig('cube_vertex', signs)
    rows = []
    start, stop = settings['fault_start'], settings['fault_end_exclusive']
    target = seed % 4
    for index, truth in enumerate(poses):
        active = regime in ('clean', 'self_occlusion') or start <= index < stop
        hand, _ = source.hand_for(truth, seed, index)
        samples = []
        obstructions = np.zeros((4, len(EDGES)), dtype=bool)
        for camera_id, original_camera in enumerate(rig):
            camera = original_camera
            captured = truth
            changed = active and target == camera_id
            obstructed = active and regime in ('external_occlusion', 'combined')
            if changed and regime in ('calibration_error', 'combined'):
                camera = replace(camera, center=camera.center + [.028, -.018, .020],
                                 rotation=rotation_vector(np.array([.035, -.046, .064])) @ camera.rotation)
            if changed and regime == 'time_skew':
                captured = poses[max(index - 4, 0)]
            geometry = source.geometry(camera, captured, hand if obstructed else FAR)
            obstructions[camera_id, geometry.edge_ids[~geometry.usable_edges]] = obstructed
            condition = 'hand' if obstructed else 'clutter' if active and regime not in ('clean', 'self_occlusion') else 'clean'
            sample = source.sample_view(geometry, source.noise_for(seed, index, camera_id), condition)
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, camera_id, 714]))
            if active and regime in ('fragmentation', 'combined'):
                sample = fragmentation(sample, rng)
            if active and regime in ('wrong_edge_ids', 'combined') and camera_id in (target, (target + 1) % 4):
                sample = decoy_edges(sample, camera, captured, rng)
            if active and regime == 'corner_dropout':
                sample = sample_change(sample, points=False)
            if active and regime == 'parallel_edges':
                labels = sample.edge_labels
                axis = np.argmax(np.abs(source.BOX.vertices[EDGES[:, 1]] - source.BOX.vertices[EDGES[:, 0]]), axis=1)
                keep = (labels >= 0) & (axis[np.maximum(labels, 0)] == 0)
                sample = sample_change(sample, points=False, segments=sample.detections.segments[keep], labels=labels[keep])
            if active and (regime == 'reacquisition' or (regime == 'single_camera_fault' and camera_id == target) or
                           (regime == 'multiple_camera_faults' and camera_id in (target, (target + 1) % 4)) or
                           (regime == 'one_view' and camera_id != target) or
                           (regime == 'combined' and camera_id == (target + 2) % 4)):
                sample = source.EMPTY
            samples.append(sample)
        rows.append(Frame(truth, tuple(samples), obstructions, active))
    return rig, initial, rows
