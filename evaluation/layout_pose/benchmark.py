"""CPU-only paired benchmark of eight diagonal camera directions at two distances."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path

# Limit native numerical threads in this process before importing NumPy.
THREAD_LIMITS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENCV_FOR_THREADS_NUM")
for variable in THREAD_LIMITS:
    os.environ[variable] = "1"

import numpy as np

from pose.fusion import (
    EDGES, SIGNS, BoxModel, Camera, Detections, Pose, associate_view, build_problem, project,
    rotation_vector, solve_pose, view_match, visible,
)


HALF_SIDE = 0.65
BOX = BoxModel((0.19, 0.12, 0.085))
INTRINSIC = np.array([[900., 0., 480.], [0., 900., 360.], [0., 0., 1.]])
IMAGE_SIZE = (960, 720)
VIEWS = ("front", "side", "top", "diagonal")
AXIS_DIRECTIONS = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
DIAGONALS = tuple(product((-1, 1), repeat=3))
HAND_DIRECTIONS = np.concatenate((AXIS_DIRECTIONS, -AXIS_DIRECTIONS, SIGNS / np.sqrt(3)))
CONDITIONS = ("clean", "clutter", "hand", *(f"missing_{name}" for name in VIEWS))
METHODS = ("corner", "edge", "hybrid")
SEEDS = (13, 57, 92)
HAND_RADIUS = 0.085
FRAME_SAMPLES = 17
FAILURE_MM, FAILURE_DEG = 20., 8.


@dataclass(frozen=True)
class Noise:
    corner_pixels: np.ndarray
    edge_pixels: np.ndarray
    keep_corners: np.ndarray
    keep_edges: np.ndarray
    trim_edges: np.ndarray
    reverse_edges: np.ndarray
    corner_clutter: np.ndarray
    edge_clutter: np.ndarray
    corner_clutter_ids: np.ndarray
    edge_clutter_ids: np.ndarray
    corner_order: np.ndarray
    edge_order: np.ndarray


@dataclass(frozen=True)
class Geometry:
    corner_ids: np.ndarray
    corners: np.ndarray
    edge_ids: np.ndarray
    edges: np.ndarray
    usable_corners: np.ndarray
    usable_edges: np.ndarray
    edge_fragments: np.ndarray


@dataclass(frozen=True)
class Sample:
    detections: Detections
    corner_labels: np.ndarray
    edge_labels: np.ndarray


def layout_name(signs):
    return "".join(f"{axis}{'+' if sign > 0 else '-'}" for axis, sign in zip("xyz", signs, strict=True))


def look_at(name, center):
    forward = -center / np.linalg.norm(center)
    world_up = np.array([0., 1., 0.]) if name == "top" else np.array([0., 0., 1.])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return Camera(name, INTRINSIC, np.stack((right, down, forward)), center, IMAGE_SIZE)


def camera_rig(mode, signs):
    diagonal = HALF_SIDE * np.asarray(signs, dtype=float)
    if mode == "equal_radius":
        diagonal /= np.sqrt(3)
    elif mode != "cube_vertex":
        raise ValueError(f"Unknown distance mode: {mode}")
    centers = np.vstack((HALF_SIDE * AXIS_DIRECTIONS, diagonal))
    return tuple(look_at(name, center) for name, center in zip(VIEWS, centers, strict=True))


def trajectory(seed, count):
    seeds = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(seeds[0])
    phase = rng.uniform(-np.pi, np.pi, size=3)
    phase_step = rng.uniform(0.14, 0.20)
    translation_amplitude = rng.uniform([0.025, 0.021, 0.015], [0.047, 0.042, 0.033])
    base_rotation = rng.uniform(-0.18, 0.18, size=3)
    rotation_amplitude = rng.uniform([0.12, 0.15, 0.18], [0.26, 0.31, 0.36])
    poses = []
    for index in range(count):
        progress = index * phase_step
        if seed % 3 == 0:
            progress += 0.46 * np.sin(progress * 1.7)
        wave = progress * np.array([1., 0.83, 0.62]) + phase
        position = translation_amplitude * np.sin(wave)
        if seed % 3 == 1:
            position *= np.cos(progress * 0.32)
        rotation = base_rotation + rotation_amplitude * np.cos(wave[::-1] + progress * 0.18)
        poses.append(Pose(rotation_vector(rotation), position))
    first = poses[0]
    prior_rng = np.random.default_rng(seeds[1])
    prior = Pose(rotation_vector(prior_rng.normal(0, 0.031, 3)) @ first.rotation,
                 first.translation + prior_rng.normal(0, 0.005, 3))
    return poses, prior


def noise_for(seed, frame, camera):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 420, frame, camera]))
    return Noise(
        rng.normal(size=(8, 2)), rng.normal(size=(12, 2, 2)), rng.random(8), rng.random(12),
        rng.uniform(0.04, 0.27, (12, 2, 1)), rng.random(12) < 0.5, rng.normal(0, 16, (3, 2)),
        rng.normal(0, 13, (3, 2, 2)), rng.permutation(8), rng.permutation(12), rng.random(11), rng.random(15),
    )


def sphere_blocks(camera_center, points_world, sphere_center, radius=HAND_RADIUS):
    """Test whether each finite camera-to-feature segment intersects a real sphere."""
    ray = points_world - camera_center
    center_offset = sphere_center - camera_center
    fraction = np.clip((ray @ center_offset) / np.maximum(np.einsum("...i,...i->...", ray, ray), 1e-12), 0, 1)
    closest = camera_center + fraction[..., None] * ray
    return np.linalg.norm(closest - sphere_center, axis=-1) < radius


def hand_for(pose, seed, frame):
    direction = HAND_DIRECTIONS[(frame // 2 + seed % len(HAND_DIRECTIONS)) % len(HAND_DIRECTIONS)]
    reach = float(np.max((BOX.vertices @ pose.rotation.T) @ direction))
    gap = 0.008 + 0.037 * (1 + np.sin(0.73 * frame + seed)) / 2
    center = pose.translation + (HAND_RADIUS + reach + gap) * direction
    return center, direction


def longest_visible_run(mask):
    transitions = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, stops = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    if not len(starts):
        return None
    at = int(np.argmax(stops - starts))
    return int(starts[at]), int(stops[at] - 1)


def geometry(camera, pose, sphere_center):
    corner_ids, corners, edge_ids, edges = visible(BOX, pose, camera)
    vertices = BOX.vertices
    world_corners = vertices[corner_ids] @ pose.rotation.T + pose.translation
    usable_corners = ~sphere_blocks(camera.center, world_corners, sphere_center)
    edge_points = vertices[EDGES[edge_ids]]
    along = np.linspace(0, 1, FRAME_SAMPLES)[None, :, None]
    samples = edge_points[:, :1] * (1 - along) + edge_points[:, 1:] * along
    world_samples = samples @ pose.rotation.T + pose.translation
    blocked = sphere_blocks(camera.center, world_samples, sphere_center)
    usable_edges = np.ones(len(edge_ids), dtype=bool)
    fragments = edges.copy()
    for row in np.flatnonzero(np.any(blocked, axis=1)):
        run = longest_visible_run(~blocked[row])
        if run is None or run[0] == run[1]:
            usable_edges[row] = False
            continue
        fragments[row] = project(samples[row, list(run)], pose, camera)[0]
        usable_edges[row] = np.linalg.norm(fragments[row, 1] - fragments[row, 0]) >= 8
    return Geometry(corner_ids, corners, edge_ids, edges, usable_corners, usable_edges, fragments)


def shuffled_features(values, labels, order):
    permutation = np.argsort(order, kind="stable")
    return values[permutation], labels[permutation]


def sample_view(data, draw, condition):
    hand = condition == "hand"
    clean = condition == "clean"
    point_keep = draw.keep_corners[data.corner_ids] >= (0.025 if clean else 0.16)
    edge_keep = draw.keep_edges[data.edge_ids] >= (0.05 if clean else 0.23)
    if hand:
        point_keep &= data.usable_corners
        edge_keep &= data.usable_edges
    point_ids = data.corner_ids[point_keep]
    edge_ids = data.edge_ids[edge_keep]
    corners = data.corners[point_keep] + (0.6 if clean else 1.25) * draw.corner_pixels[point_ids]
    edges = (data.edge_fragments if hand else data.edges)[edge_keep]
    edges = edges + draw.trim_edges[edge_ids] * (edges[:, ::-1] - edges)
    edges += (0.5 if clean else 0.95) * draw.edge_pixels[edge_ids]
    reverse = draw.reverse_edges[edge_ids]
    edges[reverse] = edges[reverse, ::-1]
    point_order = draw.corner_order[point_ids]
    edge_order = draw.edge_order[edge_ids]
    if not clean:
        point_lookup = {ident: px for ident, px in zip(data.corner_ids, data.corners, strict=True)}
        edge_lookup = {ident: px for ident, px in zip(data.edge_ids, data.edges, strict=True)}
        point_clutter_ids = [ident for ident in draw.corner_clutter_ids if ident in point_lookup][:3]
        edge_clutter_ids = [ident for ident in draw.edge_clutter_ids if ident in edge_lookup][:3]
        if point_clutter_ids:
            count = len(point_clutter_ids)
            corners = np.concatenate((corners, np.stack([point_lookup[ident] for ident in point_clutter_ids])
                                      + draw.corner_clutter[:count]))
            point_ids = np.r_[point_ids, np.full(count, -1)]
            point_order = np.r_[point_order, draw.corner_order[8:8 + count]]
        if edge_clutter_ids:
            count = len(edge_clutter_ids)
            clutter = np.stack([edge_lookup[ident] for ident in edge_clutter_ids])
            clutter += 0.18 * (clutter[:, ::-1] - clutter)
            edges = np.concatenate((edges, clutter + draw.edge_clutter[:count]))
            edge_ids = np.r_[edge_ids, np.full(count, -1)]
            edge_order = np.r_[edge_order, draw.edge_order[12:12 + count]]
    corners, point_ids = shuffled_features(corners, point_ids, point_order)
    edges, edge_ids = shuffled_features(edges, edge_ids, edge_order)
    return Sample(Detections(corners, edges), point_ids, edge_ids)


EMPTY = Sample(Detections(np.empty((0, 2)), np.empty((0, 2, 2))), np.empty(0, int), np.empty(0, int))


def fit_frame(prior, rig, detections, method):
    """All inputs are calibrated camera geometry, image detections and the causal prior."""
    matches = []
    for camera, detected in zip(rig, detections, strict=True):
        points, edges = associate_view(prior, camera, detected, BOX)
        if method == "corner":
            edges = edges[:0]
        if method == "edge":
            points = points[:0]
        matches.append(view_match(camera, detected, points, edges))
    return solve_pose(prior, build_problem(BOX, matches)), matches


def evaluate_edges(samples, matches, geometry_by_camera, designated, condition):
    observed = np.zeros((4, len(EDGES)), dtype=bool)
    correct = np.zeros_like(observed)
    geometric = np.zeros_like(observed)
    pre_hand = np.zeros_like(observed)
    claims = np.full(observed.shape, -2, dtype=int)
    for index, (sample, match, data) in enumerate(zip(samples, matches, geometry_by_camera, strict=True)):
        observed[index, sample.edge_labels[sample.edge_labels >= 0]] = True
        pre_hand[index, data.edge_ids] = True
        if condition == "hand":
            geometric[index, data.edge_ids[data.usable_edges]] = True
        elif not condition == f"missing_{VIEWS[index]}":
            geometric[index, data.edge_ids] = True
        model, observation = match.segment_pairs.T
        claimed_labels = sample.edge_labels[observation]
        claims[index, model] = claimed_labels
        correct[index, model[model == claimed_labels]] = True
    views_geometric = geometric.sum(axis=0)
    views_detected = observed.sum(axis=0)
    views_correct = correct.sum(axis=0)
    paired_claims = correct_pair_claims = 0
    for first, second in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        both = (claims[first] != -2) & (claims[second] != -2)
        paired_claims += np.count_nonzero(both)
        correct_pair_claims += np.count_nonzero(both & correct[first] & correct[second])
    blocked = np.zeros(len(EDGES), dtype=bool)
    geometric_elsewhere = detected_elsewhere = correct_elsewhere = 0
    if designated is not None:
        blocked = pre_hand[designated] & ~geometric[designated]
        other = np.arange(4) != designated
        geometric_elsewhere = np.count_nonzero(blocked & np.any(geometric[other], axis=0))
        detected_elsewhere = np.count_nonzero(blocked & np.any(observed[other], axis=0))
        correct_elsewhere = np.count_nonzero(blocked & np.any(correct[other], axis=0))
    return dict(
        geometric_edges_1plus=np.count_nonzero(views_geometric >= 1),
        geometric_edges_2plus=np.count_nonzero(views_geometric >= 2),
        detected_edges_1plus=np.count_nonzero(views_detected >= 1),
        detected_edges_2plus=np.count_nonzero(views_detected >= 2),
        correct_edges_2plus=np.count_nonzero(views_correct >= 2),
        edges_assigned=np.count_nonzero(claims != -2), edges_correct=np.count_nonzero(correct),
        multiview_pairs=paired_claims, multiview_pairs_correct=correct_pair_claims,
        designated_edges_lost=np.count_nonzero(blocked), designated_edges_visible_elsewhere=geometric_elsewhere,
        designated_edges_detected_elsewhere=detected_elsewhere, designated_edges_correct_elsewhere=correct_elsewhere,
    )


def errors(estimated, truth):
    mm = 1000 * np.linalg.norm(estimated.translation - truth.translation)
    cosine = (np.trace(estimated.rotation @ truth.rotation.T) - 1) / 2
    return float(mm), float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def designated_camera(condition, hand_direction, diagonal):
    if condition.startswith("missing_"):
        return VIEWS.index(condition.removeprefix("missing_"))
    if condition == "hand":
        candidates = np.vstack((AXIS_DIRECTIONS, np.asarray(diagonal) / np.sqrt(3)))
        exact = np.flatnonzero(np.isclose(candidates @ hand_direction, 1))
        if len(exact):
            return int(exact[0])
    return None


def evaluate(seed, frames, mode, diagonal):
    poses, initial = trajectory(seed, frames)
    rig = camera_rig(mode, diagonal)
    layout = layout_name(diagonal)
    priors = {(condition, method): initial for condition in CONDITIONS for method in METHODS}
    rows = []
    for frame, truth in enumerate(poses):
        sphere, direction = hand_for(truth, seed, frame)
        frame_geometry = tuple(geometry(camera, truth, sphere) for camera in rig)
        draws = tuple(noise_for(seed, frame, camera) for camera in range(4))
        measured = {condition: tuple(sample_view(data, draw, condition) for data, draw in zip(frame_geometry, draws, strict=True))
                    for condition in ("clean", "clutter", "hand")}
        for condition in CONDITIONS:
            designated = designated_camera(condition, direction, diagonal)
            if condition.startswith("missing_"):
                views = tuple(EMPTY if index == designated else view for index, view in enumerate(measured["clutter"]))
            else:
                views = measured[condition]
            detections = tuple(view.detections for view in views)
            for method in METHODS:
                key = condition, method
                fit, matches = fit_frame(priors[key], rig, detections, method)
                if fit.valid:
                    priors[key] = fit.pose
                mm, deg = errors(fit.pose, truth)
                failure = not fit.valid or mm > FAILURE_MM or deg > FAILURE_DEG
                edge_metrics = evaluate_edges(views, matches, frame_geometry, designated, condition)
                rows.append(dict(seed=seed, frame=frame, distance_mode=mode, layout=layout, condition=condition,
                                 hand_direction=" ".join(str(round(float(value), 3)) for value in direction) if condition == "hand" else "",
                                 designated_camera=VIEWS[designated] if designated is not None else "", method=method,
                                 translation_mm=mm, rotation_deg=deg, valid=fit.valid, failure=failure, **edge_metrics))
    return rows


def percentage(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def summarize(rows, key_fields):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in groups.items():
        mm = np.array([row["translation_mm"] for row in group])
        deg = np.array([row["rotation_deg"] for row in group])
        fields = (
            "detected_edges_2plus", "correct_edges_2plus", "edges_assigned", "edges_correct", "multiview_pairs",
            "multiview_pairs_correct", "designated_edges_lost", "designated_edges_visible_elsewhere",
            "designated_edges_detected_elsewhere", "designated_edges_correct_elsewhere",
        )
        counts = {field: sum(row[field] for row in group) for field in fields}
        lost = counts["designated_edges_lost"]
        summaries.append(dict(
            **dict(zip(key_fields, key, strict=True)), frames=len(group), translation_rmse_mm=np.sqrt(np.mean(mm**2)),
            rotation_rmse_deg=np.sqrt(np.mean(deg**2)), translation_p95_mm=np.percentile(mm, 95),
            rotation_p95_deg=np.percentile(deg, 95), failure_pct=100 * np.mean([row["failure"] for row in group]),
            invalid_pct=100 * np.mean([not row["valid"] for row in group]),
            mean_geometric_edges_1plus=np.mean([row["geometric_edges_1plus"] for row in group]),
            mean_geometric_edges_2plus=np.mean([row["geometric_edges_2plus"] for row in group]),
            mean_detected_edges_1plus=np.mean([row["detected_edges_1plus"] for row in group]),
            mean_detected_edges_2plus=counts["detected_edges_2plus"] / len(group),
            mean_correct_edges_2plus=counts["correct_edges_2plus"] / len(group),
            edge_id_precision_pct=percentage(counts["edges_correct"], counts["edges_assigned"]),
            pair_edge_id_precision_pct=percentage(counts["multiview_pairs_correct"], counts["multiview_pairs"]),
            detected_multiview_edge_recall_pct=percentage(counts["correct_edges_2plus"], counts["detected_edges_2plus"]),
            rescue_geometric_pct=percentage(counts["designated_edges_visible_elsewhere"], lost),
            rescue_detected_pct=percentage(counts["designated_edges_detected_elsewhere"], lost),
            rescue_correct_pct=percentage(counts["designated_edges_correct_elsewhere"], lost), **counts,
        ))
    return summaries


def compare_modes(by_condition):
    rows = []
    equal = {(row["layout"], row["condition"], row["method"]): row for row in by_condition
             if row["distance_mode"] == "equal_radius"}
    fields = ("translation_rmse_mm", "rotation_rmse_deg", "translation_p95_mm", "rotation_p95_deg", "failure_pct",
              "mean_geometric_edges_2plus", "pair_edge_id_precision_pct", "rescue_correct_pct")
    for row in by_condition:
        if row["distance_mode"] != "cube_vertex":
            continue
        key = row["layout"], row["condition"], row["method"]
        first = equal[key]
        deltas = {f"cube_minus_equal_{field}": (float(row[field] - first[field])
                  if first[field] is not None and row[field] is not None else None) for field in fields}
        rows.append(dict(layout=key[0], condition=key[1], method=key[2], **deltas))
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=28, help="28 frames cover all 14 hand approach directions")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    if args.frames < 1 or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("use positive frames and distinct nonnegative seeds")
    rows = []
    for mode in ("equal_radius", "cube_vertex"):
        for signs in DIAGONALS:
            for seed in args.seeds:
                rows.extend(evaluate(seed, args.frames, mode, signs))
            print(f"Completed {mode} {layout_name(signs)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_condition = summarize(rows, ("distance_mode", "layout", "condition", "method"))
    by_layout = summarize(rows, ("distance_mode", "layout", "method"))
    affected = [row for row in rows if row["designated_camera"]]
    by_camera = summarize(affected, ("distance_mode", "layout", "condition", "designated_camera", "method"))
    hand = summarize([row for row in rows if row["condition"] == "hand"],
                     ("distance_mode", "layout", "hand_direction", "method"))
    for file, measures in (("frame_metrics.csv", rows), ("by_condition.csv", by_condition), ("by_layout.csv", by_layout),
                           ("by_designated_camera.csv", by_camera), ("by_hand_direction.csv", hand),
                           ("distance_delta.csv", compare_modes(by_condition))):
        if measures:
            write_csv(args.output_dir / file, measures)
    manifest = {
        "units": "metres/degrees/pixels", "fixed_centers": {name: (HALF_SIDE * axis).tolist()
            for name, axis in zip(VIEWS, AXIS_DIRECTIONS)}, "half_side_m": HALF_SIDE, "box_size_m": BOX.size_m,
        "intrinsics": INTRINSIC.tolist(), "image_width_height": IMAGE_SIZE, "hand_sphere_radius_m": HAND_RADIUS,
        "diagonal_centers": {mode: {layout_name(signs): camera_rig(mode, signs)[-1].center.tolist() for signs in DIAGONALS}
                             for mode in ("equal_radius", "cube_vertex")},
        "frames_per_trial": args.frames, "seeds": args.seeds, "methods": METHODS, "conditions": CONDITIONS,
        "failure_mm_deg": [FAILURE_MM, FAILURE_DEG], "native_threads": {name: os.environ[name] for name in THREAD_LIMITS},
        "prior_error_mm_deg": {seed: errors(trajectory(seed, args.frames)[1], trajectory(seed, args.frames)[0][0])
                               for seed in args.seeds},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} pose rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
