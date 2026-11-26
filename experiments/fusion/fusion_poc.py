"""Reproducible calibrated multi-view cuboid tracking with synthetic detections."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np


WIDTH, HEIGHT = 960, 720
FOCAL = 850.0
CENTER = np.array([WIDTH / 2, HEIGHT / 2])
HALF_SIZE = np.array([0.095, 0.06, 0.0425])  # Metres in the object frame.
RIG_HALF_SIDE = 0.65  # Metres between the center and each face of the cube.
SIGNS = np.array(list(product((-1, 1), repeat=3)))
VERTICES = SIGNS * HALF_SIZE
EDGES = np.array([
    (a, b) for a in range(8) for b in range(a + 1, 8)
    if np.count_nonzero(SIGNS[a] != SIGNS[b]) == 1
])
EDGE_AXES = np.argmax(SIGNS[EDGES[:, 0]] != SIGNS[EDGES[:, 1]], axis=1)
METHODS = ("point", "edge", "hybrid")
LAYOUTS = ("same", "opposite")
PIXEL_GATE = 22.0
FAILURE_MM = 20.0
FAILURE_DEG = 8.0


@dataclass(frozen=True)
class Pose:
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True)
class Camera:
    name: str
    center: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class Detections:
    points: np.ndarray
    segments: np.ndarray


@dataclass(frozen=True)
class Labels:
    point_ids: np.ndarray
    edge_ids: np.ndarray


@dataclass(frozen=True)
class Observation:
    camera: Camera
    vertex_ids: np.ndarray
    pixels: np.ndarray
    normals: np.ndarray | None


@dataclass(frozen=True)
class Scenario:
    point_noise: float
    edge_noise: float
    point_dropout: float
    edge_dropout: float
    point_clutter: int
    edge_clutter: int


SCENARIOS = {
    "baseline": Scenario(0.8, 0.55, 0.08, 0.12, 1, 1),
    "stress": Scenario(1.7, 1.2, 0.45, 0.30, 3, 3),
}


def skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def rotation_vector(vector):
    angle = np.linalg.norm(vector)
    cross = skew(vector)
    if angle < 1e-8:
        return np.eye(3) + cross + 0.5 * cross @ cross
    return np.eye(3) + np.sin(angle) / angle * cross + (1 - np.cos(angle)) / angle**2 * cross @ cross


def increment(pose, delta):
    angle = np.linalg.norm(delta[:3])
    cross = skew(delta[:3])
    if angle < 1e-8:
        translation_jacobian = np.eye(3) + 0.5 * cross + (cross @ cross) / 6
    else:
        translation_jacobian = (
            np.eye(3) + (1 - np.cos(angle)) / angle**2 * cross
            + (angle - np.sin(angle)) / angle**3 * cross @ cross
        )
    rotation = rotation_vector(delta[:3])
    return Pose(rotation @ pose.rotation, rotation @ pose.translation + translation_jacobian @ delta[3:])


def cameras(layout="same"):
    diagonal = np.full(3, RIG_HALF_SIDE)
    if layout == "opposite":
        diagonal = -diagonal
    rig = []
    for name, position, up in (
        ("front", [0, 0, RIG_HALF_SIDE], [0, 1, 0]),
        ("side", [RIG_HALF_SIDE, 0, 0], [0, 1, 0]),
        ("top", [0, RIG_HALF_SIDE, 0], [0, 0, -1]),
        ("diagonal", diagonal, [0, 1, 0]),
    ):
        center = np.asarray(position, dtype=float)
        forward = -center / np.linalg.norm(center)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rig.append(Camera(name, center, np.stack((right, down, forward))))
    return tuple(rig)


def trajectory(frame):
    phase = frame * 0.065
    angles = np.array([0.10 + 0.14 * np.sin(phase), -0.19 + 0.18 * np.cos(phase * 0.79),
                       0.09 + 0.17 * np.sin(phase * 0.71)])
    translation = np.array([0.047 * np.sin(phase), 0.036 * np.sin(1.14 * phase + 0.2),
                            0.025 * np.cos(1.02 * phase)])
    return Pose(rotation_vector(angles), translation)


def initial_pose():
    # Fixed approximate hand-supplied fixture setup. No simulator pose is queried.
    return Pose(rotation_vector(np.array([0.045, 0.035, 0.12])), np.array([0.008, 0.002, 0.020]))


def project(pose, camera, vertex_ids, jacobian=False):
    world = VERTICES[vertex_ids] @ pose.rotation.T + pose.translation
    camera_xyz = (world - camera.center) @ camera.rotation.T
    depth = camera_xyz[:, 2]
    pixels = CENTER + FOCAL * camera_xyz[:, :2] / depth[:, None]
    if not jacobian:
        return pixels, depth
    normalized = camera_xyz[:, :2] / depth[:, None]
    differential = np.repeat(np.eye(2, 3)[None], len(world), axis=0)
    differential[:, :, 2] = -normalized
    differential *= (FOCAL / depth)[:, None, None]
    motion = np.concatenate((-np.stack([skew(point) for point in world]), np.broadcast_to(
        np.eye(3), (len(world), 3, 3))), axis=2)
    return pixels, differential @ camera.rotation @ motion


def visible_projections(pose, camera):
    pixels, depth = project(pose, camera, np.arange(8))
    inside = (depth > 0) & np.all((pixels >= 0) & (pixels < [WIDTH, HEIGHT]), axis=1)
    local_camera = pose.rotation.T @ (camera.center - pose.translation)
    front_faces = SIGNS * local_camera > HALF_SIZE
    point_ids = np.flatnonzero(inside & np.any(front_faces, axis=1))
    edge_faces = front_faces[EDGES[:, 0]].copy()
    edge_faces[np.arange(len(EDGES)), EDGE_AXES] = False
    segments = pixels[EDGES]
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    edge_ids = np.flatnonzero(np.all(inside[EDGES], axis=1) & np.any(edge_faces, axis=1) & (lengths >= 10))
    return point_ids, pixels[point_ids], edge_ids, segments[edge_ids]


def shuffle_detections(values, labels, rng):
    order = rng.permutation(len(values))
    return values[order], np.asarray(labels, dtype=int)[order]


def simulate(pose, camera, scenario, scenario_name, frame, rng):
    point_ids, points, edge_ids, segments = visible_projections(pose, camera)
    point_keep = rng.random(len(points)) >= scenario.point_dropout
    edge_keep = rng.random(len(segments)) >= scenario.edge_dropout
    if scenario_name == "stress" and 26 <= frame < 34:
        point_keep[:] = False
    if scenario_name == "stress" and 48 <= frame < 54 and camera.name == "diagonal":
        point_keep[:] = False
        edge_keep[:] = False
        return Detections(np.empty((0, 2)), np.empty((0, 2, 2))), Labels(np.zeros(0, int), np.zeros(0, int))

    measured_points = points[point_keep] + rng.normal(0, scenario.point_noise, (point_keep.sum(), 2))
    point_clutter = points[rng.integers(len(points), size=scenario.point_clutter)] + rng.normal(
        0, 14, (scenario.point_clutter, 2)
    )
    measured_points, point_labels = shuffle_detections(
        np.concatenate((measured_points, point_clutter)),
        np.concatenate((point_ids[point_keep], np.full(scenario.point_clutter, -1))), rng,
    )

    # Detected segment endpoints can sit anywhere along the model edge.
    trim = rng.uniform(0.04, 0.30, (len(segments), 2, 1))
    trimmed = segments + trim * (segments[:, ::-1] - segments)
    measured_edges = trimmed[edge_keep] + rng.normal(0, scenario.edge_noise, (edge_keep.sum(), 2, 2))
    edge_clutter = trimmed[rng.integers(len(segments), size=scenario.edge_clutter)] + rng.normal(
        0, 12, (scenario.edge_clutter, 2, 2)
    )
    edges = np.concatenate((measured_edges, edge_clutter))
    order_flip = rng.random(len(edges)) < 0.5
    edges[order_flip] = edges[order_flip, ::-1]
    edges, edge_labels = shuffle_detections(
        edges, np.concatenate((edge_ids[edge_keep], np.full(scenario.edge_clutter, -1))), rng,
    )
    return Detections(measured_points, edges), Labels(point_labels, edge_labels)


def simulate_frame(truth, rig, scenario_name, seed, frame):
    # Axis cameras receive exactly the same samples in both layouts. The
    # diagonals also receive the same random stream for each frame.
    return [
        simulate(truth, camera, SCENARIOS[scenario_name], scenario_name, frame,
                 np.random.default_rng(np.random.SeedSequence([seed, frame, index])))
        for index, camera in enumerate(rig)
    ]


def associate(costs, gate):
    """Use gated greedy one-to-one matching, lowest finite cost first."""
    pairs = np.argwhere(costs < gate)
    if not len(pairs):
        return np.empty((0, 2), dtype=int)
    order = np.argsort(costs[pairs[:, 0], pairs[:, 1]], kind="stable")
    used_models, used_detections, matches = set(), set(), []
    for model, detection in pairs[order]:
        if model not in used_models and detection not in used_detections:
            matches.append((model, detection))
            used_models.add(model)
            used_detections.add(detection)
    return np.asarray(matches)


def line_geometry(segments):
    difference = segments[:, 1] - segments[:, 0]
    lengths = np.linalg.norm(difference, axis=1)
    directions = difference / np.maximum(lengths[:, None], 1e-8)
    return directions, lengths


def match_edges(predicted, detected):
    model_dirs, model_lengths = line_geometry(predicted)
    obs_dirs, obs_lengths = line_geometry(detected)
    midpoints = detected.mean(axis=1)[None, :, :] - predicted.mean(axis=1)[:, None, :]
    along = np.abs(np.einsum("ijk,ik->ij", midpoints, model_dirs))
    normals = model_dirs[:, ::-1] * [-1, 1]
    across = np.abs(np.einsum("ijk,ik->ij", midpoints, normals))
    angular = np.arccos(np.clip(np.abs(model_dirs @ obs_dirs.T), 0, 1))
    relative_length = obs_lengths[None] / model_lengths[:, None]
    cost = across + 0.30 * along + 22 * angular
    cost[(angular > np.deg2rad(28)) | (relative_length < 0.30) | (relative_length > 1.35)
         | (along > 0.42 * model_lengths[:, None]) | (obs_lengths[None] < 7)] = np.inf
    return associate(cost, PIXEL_GATE)


def estimated_associations(prior, camera, detections):
    """Return model ID/detection index pairs using pose, calibration and pixels only."""
    point_ids, points, edge_ids, segments = visible_projections(prior, camera)
    point_distances = np.linalg.norm(points[:, None] - detections.points, axis=2)
    point_pairs = associate(point_distances, PIXEL_GATE)
    edge_pairs = match_edges(segments, detections.segments)
    return (np.column_stack((point_ids[point_pairs[:, 0]], point_pairs[:, 1])),
            np.column_stack((edge_ids[edge_pairs[:, 0]], edge_pairs[:, 1])))


def oracle_associations(labels):
    point_rows = np.flatnonzero(labels.point_ids >= 0)
    edge_rows = np.flatnonzero(labels.edge_ids >= 0)
    return (np.column_stack((labels.point_ids[point_rows], point_rows)),
            np.column_stack((labels.edge_ids[edge_rows], edge_rows)))


def observations(camera, detections, point_pairs, edge_pairs):
    result = []
    if len(point_pairs):
        model, rows = point_pairs.T
        result.append(Observation(camera, model, detections.points[rows], None))
    if len(edge_pairs):
        model, rows = edge_pairs.T
        segments = detections.segments[rows]
        directions, _ = line_geometry(segments)
        normals = directions[:, ::-1] * [-1, 1]
        result.append(Observation(camera, EDGES[model].ravel(), np.repeat(segments.mean(axis=1), 2, axis=0),
                                  np.repeat(normals, 2, axis=0)))
    return result


def residual_and_jacobian(pose, inputs):
    residuals, derivatives = [], []
    for observation in inputs:
        pixels, jacobian = project(pose, observation.camera, observation.vertex_ids, jacobian=True)
        error = pixels - observation.pixels
        if observation.normals is not None:
            error = np.einsum("ij,ij->i", observation.normals, error)
            jacobian = np.einsum("ij,ijk->ik", observation.normals, jacobian)
        residuals.append(error.ravel())
        derivatives.append(jacobian.reshape(-1, 6))
    return np.concatenate(residuals), np.concatenate(derivatives)


def robust_cost(residual):
    magnitude = np.abs(residual)
    return np.where(magnitude <= 3.0, 0.5 * residual**2, 3.0 * (magnitude - 1.5)).sum()


def fit_pose(prior, inputs):
    if not inputs:
        return prior, False
    residual, jacobian = residual_and_jacobian(prior, inputs)
    if np.linalg.matrix_rank(jacobian, tol=1e-5) < 6:
        return prior, False
    pose, damping = prior, 1e-3
    for _ in range(12):
        weight = np.minimum(1.0, 3.0 / np.maximum(np.abs(residual), 1e-8))
        normal = jacobian.T @ (weight[:, None] * jacobian)
        gradient = jacobian.T @ (weight * residual)
        delta = np.linalg.solve(normal + damping * np.diag(np.maximum(np.diag(normal), 1.0)), -gradient)
        if np.linalg.norm(delta) < 1e-7:
            break
        candidate = increment(pose, delta)
        new_residual, new_jacobian = residual_and_jacobian(candidate, inputs)
        if np.all(np.isfinite(new_residual)) and robust_cost(new_residual) < robust_cost(residual):
            pose, residual, jacobian = candidate, new_residual, new_jacobian
            damping = max(damping / 3, 1e-7)
        else:
            damping *= 8
    return pose, True


def fit_frame(prior, views, method, oracle_labels=None):
    inputs, associations = [], []
    for index, (camera, detections) in enumerate(views):
        if oracle_labels is None:
            point_pairs, edge_pairs = estimated_associations(prior, camera, detections)
        else:
            point_pairs, edge_pairs = oracle_associations(oracle_labels[index])
        if method == "point":
            edge_pairs = edge_pairs[:0]
        if method == "edge":
            point_pairs = point_pairs[:0]
        inputs.extend(observations(camera, detections, point_pairs, edge_pairs))
        associations.append((point_pairs, edge_pairs))
    pose, solved = fit_pose(prior, inputs)
    return pose, solved, associations


def pose_error(estimated, truth):
    millimeters = 1000 * np.linalg.norm(estimated.translation - truth.translation)
    relative = estimated.rotation @ truth.rotation.T
    degrees = np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    return float(millimeters), float(degrees)


def frame_coverage(truth, rig, samples, scenario_name, seed, frame, layout):
    geometric = [visible_projections(truth, camera) for camera in rig]
    for feature, count, eligible, detected in (
        ("corner", len(VERTICES), [row[0] for row in geometric], [labels.point_ids for _, labels in samples]),
        ("edge", len(EDGES), [row[2] for row in geometric], [labels.edge_ids for _, labels in samples]),
    ):
        geometric_views = np.bincount(np.concatenate(eligible), minlength=count)
        detected_ids = np.concatenate(detected)
        detected_views = np.bincount(detected_ids[detected_ids >= 0], minlength=count)
        for feature_id in range(count):
            yield dict(scenario=scenario_name, seed=seed, frame=frame, layout=layout, feature=feature,
                       feature_id=feature_id, geometric_views=int(geometric_views[feature_id]),
                       detected_views=int(detected_views[feature_id]))


def run_trial(frames, seed, scenario_name, layout, axis_only=False):
    rig = cameras(layout)
    if axis_only:
        rig = rig[:3]
        settings = [("axes", "estimated", "hybrid")]
    else:
        settings = list(product(("diagonal", "four"), ("estimated", "oracle"), METHODS))
    poses = {setting: initial_pose() for setting in settings}
    records, coverage = [], []
    for frame in range(frames):
        truth = trajectory(frame)
        samples = simulate_frame(truth, rig, scenario_name, seed, frame)
        if not axis_only:
            coverage.extend(frame_coverage(truth, rig, samples, scenario_name, seed, frame, layout))
        views = tuple((camera, detections) for camera, (detections, _) in zip(rig, samples, strict=True))
        evaluation_labels = tuple(labels for _, labels in samples)
        for view_mode, association_mode, method in settings:
            key = view_mode, association_mode, method
            prior = poses[key]
            trial_views = views[-1:] if view_mode == "diagonal" else views
            labels = evaluation_labels[-1:] if view_mode == "diagonal" else evaluation_labels
            poses[key], solved, associations = fit_frame(
                prior, trial_views, method, oracle_labels=labels if association_mode == "oracle" else None
            )
            used, wrong = 0, 0
            for (point_pairs, edge_pairs), truth_labels in zip(associations, labels, strict=True):
                used += len(point_pairs) + len(edge_pairs)
                # Labels are used after matching, solely to report errors in either path.
                wrong += np.count_nonzero(truth_labels.point_ids[point_pairs[:, 1]] != point_pairs[:, 0])
                wrong += np.count_nonzero(truth_labels.edge_ids[edge_pairs[:, 1]] != edge_pairs[:, 0])
            translation, rotation = pose_error(poses[key], truth)
            records.append(dict(
                scenario=scenario_name, seed=seed, frame=frame, layout=layout, views=view_mode,
                association=association_mode,
                method=method, translation_mm=translation, rotation_deg=rotation, solved=solved,
                failed=translation > FAILURE_MM or rotation > FAILURE_DEG, matches=used, wrong_matches=int(wrong),
            ))
    return records, coverage


def summarize(records):
    groups = {}
    for row in records:
        key = tuple(row[field] for field in ("scenario", "layout", "views", "association", "method"))
        groups.setdefault(key, []).append(row)
    summary = []
    for (scenario, layout, views, association, method), rows in groups.items():
        translation = np.array([row["translation_mm"] for row in rows])
        rotation = np.array([row["rotation_deg"] for row in rows])
        matches = sum(row["matches"] for row in rows)
        wrong = sum(row["wrong_matches"] for row in rows)
        summary.append(dict(
            scenario=scenario, layout=layout, views=views, association=association, method=method, frames=len(rows),
            translation_rmse_mm=float(np.sqrt(np.mean(translation**2))),
            rotation_rmse_deg=float(np.sqrt(np.mean(rotation**2))),
            translation_p95_mm=float(np.percentile(translation, 95)), rotation_p95_deg=float(np.percentile(rotation, 95)),
            held_frames=sum(not row["solved"] for row in rows), failed_frames=sum(row["failed"] for row in rows),
            invalid_rate_pct=100 * sum(not row["solved"] for row in rows) / len(rows),
            failure_rate_pct=100 * sum(row["failed"] for row in rows) / len(rows),
            matches=matches, wrong_matches=wrong, wrong_match_pct=100 * wrong / matches if matches else None,
        ))
    return summary


def summarize_coverage(records):
    groups = {}
    frame_groups = {}
    for row in records:
        key = tuple(row[field] for field in ("scenario", "layout", "feature"))
        groups.setdefault((*key, row["feature_id"]), []).append(row)
        frame_groups.setdefault((*key, row["seed"], row["frame"]), []).append(row)
    features = []
    for (scenario, layout, feature, feature_id), rows in groups.items():
        features.append(dict(
            scenario=scenario, layout=layout, feature=feature, feature_id=feature_id,
            geometric_1plus_pct=100 * np.mean([row["geometric_views"] >= 1 for row in rows]),
            geometric_2plus_pct=100 * np.mean([row["geometric_views"] >= 2 for row in rows]),
            detected_1plus_pct=100 * np.mean([row["detected_views"] >= 1 for row in rows]),
            detected_2plus_pct=100 * np.mean([row["detected_views"] >= 2 for row in rows]),
        ))
    counts = {}
    for (scenario, layout, feature, _, _), rows in frame_groups.items():
        counts.setdefault((scenario, layout, feature), []).append([
            sum(row[field] >= threshold for row in rows)
            for field in ("geometric_views", "detected_views") for threshold in (1, 2)
        ])
    layouts = []
    for (scenario, layout, feature), per_frame in counts.items():
        visible, multiview, detected, detected_multiview = np.mean(per_frame, axis=0)
        layouts.append(dict(scenario=scenario, layout=layout, feature=feature,
                            mean_geometric_1plus=visible, mean_geometric_2plus=multiview,
                            mean_detected_1plus=detected, mean_detected_2plus=detected_multiview))
    return features, layouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42, help="First random seed")
    parser.add_argument("--trials", type=int, default=3, help="Consecutive random seeds per scenario")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--axis-only", action="store_true", help="Run the shared three-axis estimated hybrid comparator")
    args = parser.parse_args()
    if min(args.frames, args.trials) <= 0 or args.seed < 0:
        parser.error("frames and trials must be positive and seed must be nonnegative")
    records, coverage = [], []
    layouts = ("common",) if args.axis_only else LAYOUTS
    for scenario in SCENARIOS:
        for seed in range(args.seed, args.seed + args.trials):
            for layout in layouts:
                run_records, run_coverage = run_trial(args.frames, seed, scenario, layout, args.axis_only)
                records.extend(run_records)
                coverage.extend(run_coverage)
    summary = summarize(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.axis_only:
        outputs = (("axis_hybrid_frame_metrics.csv", records), ("axis_hybrid_summary.csv", summary))
    else:
        feature_summary, layout_summary = summarize_coverage(coverage)
        outputs = (("frame_metrics.csv", records), ("summary.csv", summary),
                   ("feature_coverage.csv", feature_summary), ("layout_coverage.csv", layout_summary))
    for name, rows in outputs:
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    config_name = "axis_hybrid_config.json" if args.axis_only else "config.json"
    config = dict(
        frames_per_trial=args.frames, first_seed=args.seed, trials=args.trials, failure_mm=FAILURE_MM,
        failure_deg=FAILURE_DEG, target_size_m=(2 * HALF_SIZE).tolist(), rig_side_m=2 * RIG_HALF_SIDE,
    )
    stress_windows = {"corner_dropout_all": [26, 34]}
    if not args.axis_only:
        config["diagonal_centers_m"] = {layout: cameras(layout)[-1].center.tolist() for layout in LAYOUTS}
        stress_windows["diagonal_outage"] = [48, 54]
    config["stress_frame_windows_end_exclusive"] = stress_windows
    (args.output_dir / config_name).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} frame measurements and {len(summary)} method summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
