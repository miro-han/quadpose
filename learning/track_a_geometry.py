"""Geometry-only inputs and classical pose fitting for the Track A learning pilots."""

from dataclasses import dataclass, replace

import numpy as np

from pose.fusion import EDGES, advance, associate_view, build_problem, line_geometry, solve_pose, view_match, visible
from pose.hypotheses import cross_score
from pose.refinement import Config, EdgeGate, add_fragments, associate, bundle_fragments, edge_gate, hungarian_pairs


EDGE_FIELDS = ("normal_px", "angle_degrees", "overlap_short_fraction", "length_ratio", "along_mid_px",
               "projected_length_px", "observed_length_px", "whole_segment_fraction", "associated_incident_corners",
               "near_incident_endpoint_px", "camera_depth_m", "line_endpoint_error_px")
RANK_FIELDS = ("fit_rms_px", "log_condition", "log_min_eigenvalue", "in_fit_residual_median", "in_fit_residual_max",
               "in_fit_feature_mean", "in_fit_support_count", "in_fit_stability_mean", "holdout_residual_px",
               "holdout_available", "holdout_supported", "holdout_features", "translation_delta_mm", "rotation_delta_deg")
GEOMETRY = Config()
BROAD = replace(GEOMETRY, topology=False, edge_distance_px=14, edge_angle_deg=34, edge_overlap=0.25, min_overlap_px=5.)
EMPTY = np.empty((0, 2), dtype=int)


@dataclass(frozen=True)
class Candidates:
    edge_ids: np.ndarray
    allowed: EdgeGate
    features: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    point_pairs: np.ndarray


def make_edge_candidates(prior, camera, detections, box):
    _, _, edge_ids, predicted = visible(box, prior, camera)
    observed = detections.segments
    points, _ = associate_view(prior, camera, detections, box)
    allowed = edge_gate(predicted, observed, edge_ids, BROAD)
    rows, cols = np.where(np.isfinite(allowed.cost))
    if not len(rows):
        return Candidates(edge_ids, allowed, np.empty((0, len(EDGE_FIELDS)), dtype=np.float32), rows, cols, points)
    pred, obs = predicted[rows], observed[cols]
    along, lengths = line_geometry(pred)
    obs_along, obs_lengths = line_geometry(obs)
    normals, obs_normals = along[:, ::-1] * [-1, 1], obs_along[:, ::-1] * [-1, 1]
    delta = obs - pred[:, :1]
    vector = np.einsum("nij,nj->ni", delta, along)
    projected_length = np.ptp(vector, axis=1)
    interval = allowed.intervals[rows, cols]
    overlap = np.maximum(0, interval[:, 1] - interval[:, 0])
    residual = np.mean(np.abs(np.einsum("nij,nj->ni", delta, normals)), axis=1)
    opposite = np.mean(np.abs(np.einsum("nij,nj->ni", pred - obs[:, :1], obs_normals)), axis=1)
    angle = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ni,ni->n", along, obs_along)), 0, 1)))
    vertices = EDGES[edge_ids[rows]]
    incident = np.isin(vertices, points[:, 0]).sum(axis=1)
    near = np.full(len(rows), 50.)
    if len(points):
        edge_ends_to_points = np.linalg.norm(obs[:, :, None] - detections.points[points[:, 1]][None, None], axis=3)
        same_vertex = np.isin(points[:, 0], np.unique(vertices)) & np.any(vertices[:, :, None] == points[:, 0], axis=1)
        near = np.minimum(50., np.min(np.where(same_vertex[:, None], edge_ends_to_points, np.inf), axis=(1, 2)))
    world_ends = box.vertices[vertices] @ prior.rotation.T + prior.translation
    depth = ((world_ends.mean(axis=1) - camera.center) @ camera.rotation[2]).clip(0, 3)
    feature = np.column_stack((residual, angle, overlap / np.maximum(np.minimum(lengths, projected_length), 1e-7),
                               obs_lengths / lengths, np.abs(vector.mean(axis=1) - 0.5 * lengths), lengths,
                               obs_lengths, overlap / lengths, incident, near, depth, opposite)).astype(np.float32)
    return Candidates(edge_ids, allowed, feature, rows, cols, points)


def candidate_targets(candidate, evaluation_labels):
    return (candidate.edge_ids[candidate.rows] == evaluation_labels[candidate.columns]).astype(np.float32)


def learned_pairs(candidates, model, threshold):
    scores = model.probability(candidates.features)
    raw = np.full(candidates.allowed.cost.shape, np.inf)
    raw[candidates.rows, candidates.columns] = np.where(scores >= threshold, 1 - scores, np.inf)
    gated = EdgeGate(raw, candidates.allowed.intervals)
    primary = hungarian_pairs(raw, 1 - threshold + 1e-8)
    edges = add_fragments(primary, gated, BROAD)
    edges[:, 0] = candidates.edge_ids[edges[:, 0]]
    return candidates.point_pairs, edges


def broad_pairs(candidates):
    scores = candidates.allowed.cost
    primary = hungarian_pairs(scores, BROAD.edge_distance_px + .22 * BROAD.edge_angle_deg + 5.01)
    edges = add_fragments(primary, candidates.allowed, BROAD)
    edges[:, 0] = candidates.edge_ids[edges[:, 0]]
    return candidates.point_pairs, edges


def edge_weights(candidates, model, edge_pairs):
    confidence = model.probability(candidates.features)
    mapping = {(int(candidates.edge_ids[row]), int(col)): float(probability)
               for row, col, probability in zip(candidates.rows, candidates.columns, confidence, strict=True)}
    return np.array([max(0.08, np.mean([mapping.get((int(edge), int(index)), 0.08)
                                      for ident, index in edge_pairs if ident == edge]))
                     for edge in np.unique(edge_pairs[:, 0])])


def fitted_pose(prior, cameras, detections, box, pairs, weights=None):
    matches, feature_weights = [], []
    for index, (camera, measured, (points, segments)) in enumerate(zip(cameras, detections, pairs, strict=True)):
        merged, bundled_pairs = bundle_fragments(measured, segments)
        matches.append(view_match(camera, merged, points, bundled_pairs))
        if weights is not None:
            feature_weights.append(np.repeat(np.r_[np.ones(len(points)), weights[index]], 2))
    problem = build_problem(box, matches)
    if weights is None:
        return solve_pose(prior, problem)
    return solve_weighted(prior, problem, np.concatenate(feature_weights))


def solve_weighted(prior, problem, confidence):
    """Same damped Huber update and rank policy, with a fixed weight per scalar residual."""
    from pose.fusion import LocalFit

    def cost(values):
        magnitude = np.abs(values)
        return float(np.sum(confidence * np.where(magnitude <= 3., magnitude**2 / 2, 3. * (magnitude - 1.5))))

    if len(problem.vertices) < 6:
        return LocalFit(prior, False, np.inf, np.inf)
    values, jacobian = problem.evaluate(prior, jacobian=True)
    singular = np.linalg.svd(jacobian * np.sqrt(confidence[:, None]), compute_uv=False)
    condition = singular[0] / max(singular[-1], 1e-8)
    if singular[-1] < .05 or condition > 5000:
        return LocalFit(prior, False, float(condition), np.inf)
    pose, damping = prior, 1e-3
    for _ in range(8):
        weighted = confidence * np.minimum(1, 3 / np.maximum(np.abs(values), 1e-8))
        normal = jacobian.T @ (jacobian * weighted[:, None])
        delta = np.linalg.solve(normal + damping * np.diag(np.maximum(1, np.diag(normal))),
                                -jacobian.T @ (weighted * values))
        if np.linalg.norm(delta) < 1e-6:
            break
        candidate = advance(pose, delta)
        residual = problem.evaluate(candidate)
        if cost(residual) < cost(values):
            pose, (values, jacobian) = candidate, problem.evaluate(candidate, jacobian=True)
            damping = max(damping / 3, 1e-6)
        else:
            damping *= 8
    return LocalFit(pose, True, float(condition), float(np.sqrt(np.mean(values**2))))


def geometry_pairs(prior, cameras, detections, box):
    return tuple(associate(prior, camera, measured, box, config=GEOMETRY, assignment="hungarian")
                 for camera, measured in zip(cameras, detections, strict=True))


def rank_features(matrix):
    features, valid = [], []
    for row in matrix.triples:
        inside = np.asarray(row.fit_views)
        held = row.omitted
        residuals = np.minimum(row.residuals_px[inside], 60)
        stability = np.nan_to_num(row.association_stability[inside], nan=0)
        features.append([min(row.fit.rms, 60), np.log1p(min(row.condition, 1e6)), np.log1p(row.min_eigenvalue),
                         np.median(residuals), np.max(residuals), np.mean(row.feature_counts[inside]),
                         row.supported[inside].sum(), np.mean(stability), min(row.residuals_px[held], 60),
                         np.isfinite(row.residuals_px[held]), row.supported[held], row.feature_counts[held],
                         1000 * row.translation_delta_m, row.rotation_delta_deg])
        valid.append(row.fit.valid and row.rank == 6)
    return np.asarray(features, dtype=np.float32), np.asarray(valid, dtype=bool)


def select_rank(matrix, features, valid, method, model=None):
    if not np.any(valid):
        return None
    if method == "rank_min_rms":
        cost = np.array([row.fit.rms for row in matrix.triples])
    elif method == "rank_cross":
        cost = np.array([cross_score(row) for row in matrix.triples])
    elif method in ("rank_linear", "rank_mlp"):
        cost = -model.forward(features)
    else:
        raise ValueError(f"Unknown rank method: {method}")
    cost = np.where(valid, cost, np.inf)
    return int(np.argmin(cost)) if np.isfinite(cost.min()) else None
