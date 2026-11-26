"""Bounded geometry-only association and pose refinement for a known cuboid."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from pose.fusion import (
    EDGE_AXES, EDGES, Detections, Pose, associate_view, build_problem, greedy_pairs, huber, line_geometry,
    project, skew, solve_pose, view_match, visible,
)


@dataclass(frozen=True)
class Config:
    edge_mode: str = "gated"
    topology: bool = True
    fragments: bool = True
    min_views: int = 1
    max_iterations: int = 3  # additional pose fits after the initial fit
    budget_ms: float = 100.0
    edge_distance_px: float = 8.0
    edge_angle_deg: float = 20.0
    edge_overlap: float = 0.48
    min_overlap_px: float = 6.0
    point_gate_px: float = 19.0
    topology_px: float = 6.0
    max_fragments: int = 3
    fragment_overlap: float = 0.25
    fragment_gap_px: float = 24.0
    min_eigenvalue: float = 0.75
    max_condition: float = 20000.0
    max_residual_rms_px: float = 12.0
    support_median_px: float = 6.0
    min_features_per_view: int = 3
    min_retention: float = 0.40
    unmatched_penalty: float = 16.0
    objective_tolerance: float = 1e-3
    step_mm: float = 0.20
    step_deg: float = 0.10


@dataclass(frozen=True)
class Observability:
    rank: int
    min_eigenvalue: float
    condition: float
    eigenvalues: np.ndarray
    information: np.ndarray
    weak_direction: np.ndarray  # normalized object-centred [translation, rotation]
    directional_information: np.ndarray  # diagonal in that same basis
    edge_direction_counts: np.ndarray  # counts distinct model edges in each view by local X/Y/Z axis
    support_counts: np.ndarray  # per-view [corners, distinct edges, raw edge fragments]
    residual_rms_px: float


@dataclass(frozen=True)
class Iteration:
    objective: float
    correspondence_changes: float  # symmetric difference / union
    step_mm: float
    step_deg: float
    elapsed_ms: float
    accepted: bool
    valid: bool
    point_count: int
    edge_count: int
    reason: str


@dataclass(frozen=True)
class RefinementResult:
    pose: Pose
    valid: bool
    reason: str
    diagnostics: Observability
    point_pairs: tuple[np.ndarray, ...]
    edge_pairs: tuple[np.ndarray, ...]
    verified_point_pairs: tuple[np.ndarray, ...]
    verified_edge_pairs: tuple[np.ndarray, ...]
    support: np.ndarray
    history: tuple[Iteration, ...]
    stage_ms: dict[str, float]


@dataclass(frozen=True)
class EdgeGate:
    cost: np.ndarray
    intervals: np.ndarray


@dataclass(frozen=True)
class Candidate:
    pose: Pose
    pairs: tuple
    verified_pairs: tuple
    diagnostics: Observability
    support: np.ndarray
    objective: float
    valid: bool
    reason: str


EMPTY = np.empty((0, 2), dtype=int)


def hungarian_pairs(cost, unmatched_cost):
    """Minimum-cost one-to-one assignment with a dedicated optional dummy per model row."""
    rows, real_columns = cost.shape
    if not rows or not real_columns:
        return EMPTY.copy()
    columns = real_columns + rows
    forbidden = max(1., unmatched_cost) * 1e6
    expanded = np.full((rows, columns), forbidden)
    expanded[:, :real_columns] = np.where(np.isfinite(cost) & (cost < unmatched_cost), cost, forbidden)
    expanded[np.arange(rows), real_columns + np.arange(rows)] = unmatched_cost
    u, v = np.zeros(rows + 1), np.zeros(columns + 1)
    parent, previous = np.zeros(columns + 1, int), np.zeros(columns + 1, int)
    for row in range(1, rows + 1):
        parent[0] = row
        unused = np.ones(columns + 1, bool)
        minimum = np.full(columns + 1, np.inf)
        at = 0
        while True:
            unused[at] = False
            active = parent[at]
            reduced = expanded[active - 1] - u[active] - v[1:]
            better = unused[1:] & (reduced < minimum[1:])
            minimum[1:][better] = reduced[better]
            previous[1:][better] = at
            next_column = int(np.argmin(np.where(unused, minimum, np.inf)))
            change = minimum[next_column]
            used_columns = np.flatnonzero(~unused)
            u[parent[used_columns]] += change
            v[used_columns] -= change
            minimum[unused] -= change
            at = next_column
            if parent[at] == 0:
                break
        while at:
            prior_column = previous[at]
            parent[at] = parent[prior_column]
            at = prior_column
    assigned_columns = np.flatnonzero(parent[1:real_columns + 1])
    matches = np.column_stack((parent[assigned_columns + 1] - 1, assigned_columns))
    return matches[np.argsort(matches[:, 0], kind="stable")]


def match(cost, unmatched_cost, assignment):
    if assignment == "greedy":
        return greedy_pairs(cost, unmatched_cost)
    if assignment == "hungarian":
        return hungarian_pairs(cost, unmatched_cost)
    raise ValueError(f"Unknown assignment: {assignment}")


def edge_gate(predicted, observed, model_ids, config, topology_points=None, *, verification=False):
    """Score only finite, direction-compatible observed intervals on a visible model edge."""
    if not len(predicted) or not len(observed):
        return EdgeGate(np.full((len(predicted), len(observed)), np.inf), np.zeros((len(predicted), len(observed), 2)))
    directions, lengths = line_geometry(predicted)
    obs_direction, obs_lengths = line_geometry(observed)
    displacement = observed[None] - predicted[:, None, :1]
    along = np.einsum("emnd,ed->emn", displacement, directions)
    normals = directions[:, ::-1] * [-1, 1]
    distance = np.mean(np.abs(np.einsum("emnd,ed->emn", displacement, normals)), axis=2)
    angle = np.degrees(np.arccos(np.clip(np.abs(directions @ obs_direction.T), 0, 1)))
    lo, hi = np.sort(along, axis=2).transpose(2, 0, 1)
    interval = np.stack((np.maximum(lo, 0), np.minimum(hi, lengths[:, None])), axis=2)
    overlap = np.maximum(0, interval[..., 1] - interval[..., 0])
    relative = overlap / np.maximum(np.minimum(lengths[:, None], hi - lo), 1e-8)
    distance_limit = config.edge_distance_px * (0.8 if verification else 1.)
    angle_limit = config.edge_angle_deg * (0.85 if verification else 1.)
    rejected = ((distance > distance_limit) | (angle > angle_limit) | (relative < config.edge_overlap)
                | (overlap < config.min_overlap_px) | (obs_lengths[None] < config.min_overlap_px))
    scores = distance + 0.22 * angle + 5 * (1 - relative)
    if config.topology and topology_points is not None and len(topology_points[0]):
        point_models, point_pixels = topology_points
        distances = np.linalg.norm(observed[:, :, None] - point_pixels[None, None], axis=-1)
        nearest = np.argmin(distances, axis=2)
        known = np.take_along_axis(distances, nearest[:, :, None], axis=2)[..., 0] <= config.topology_px
        adjacent = np.any(point_models[nearest][None, :, :, None] == EDGES[model_ids][:, None, None], axis=3)
        distance_to_end = np.min(np.linalg.norm(predicted[:, None, :, None] - observed[None, :, None], axis=4), axis=2)
        near_end = distance_to_end <= 1.5 * config.topology_px
        rejected |= np.any(known[None] & near_end & ~adjacent, axis=2)
        scores -= 1.5 * np.count_nonzero(known[None] & near_end & adjacent, axis=2)
    scores[rejected | ~np.isfinite(scores)] = np.inf
    return EdgeGate(scores, interval)


def add_fragments(primary, allowed, config):
    if not config.fragments or not len(primary):
        return primary
    groups = {int(model): [int(raw)] for model, raw in primary}
    used = set(primary[:, 1])
    candidates = sorted((allowed.cost[model, raw], model, raw) for model in groups
                        for raw in np.flatnonzero(np.isfinite(allowed.cost[model])) if raw not in used)
    for _, model, raw in candidates:
        if raw in used or len(groups[model]) >= config.max_fragments:
            continue
        low, high = allowed.intervals[model, raw]
        chosen = allowed.intervals[model, groups[model]]
        overlap = np.maximum(0, np.minimum(high, chosen[:, 1]) - np.maximum(low, chosen[:, 0]))
        gap = np.maximum(0, np.maximum(low, chosen[:, 0]) - np.minimum(high, chosen[:, 1]))
        permitted = config.fragment_overlap * np.minimum(high - low, chosen[:, 1] - chosen[:, 0])
        if np.all(overlap <= permitted) and np.min(gap) <= config.fragment_gap_px:
            groups[model].append(int(raw))
            used.add(raw)
    return np.array([(model, raw) for model, observations in groups.items() for raw in observations], dtype=int)


def topology_points(pairs, prior, camera, detections, box, config):
    model, raw = pairs.T
    pixels = detections.points[raw]
    projected = project(box.vertices[model], prior, camera)[0]
    confident = np.linalg.norm(projected - pixels, axis=1) <= config.topology_px
    return model[confident], pixels[confident]


def associate(prior, camera, detections, box, *, config=None, assignment="greedy"):
    """Return (model ID, original observation index) pairs for each feature type."""
    config = Config() if config is None else config
    if config.edge_mode == "legacy" and assignment == "greedy":
        return associate_view(prior, camera, detections, box)
    point_ids, point_pixels, edge_ids, segments = visible(box, prior, camera)
    point_cost = np.linalg.norm(point_pixels[:, None] - detections.points[None], axis=2)
    point_pairs = match(point_cost, config.point_gate_px, assignment)
    point_pairs[:, 0] = point_ids[point_pairs[:, 0]]
    if config.edge_mode == "legacy":
        return point_pairs, associate_view(prior, camera, detections, box)[1]
    landmarks = topology_points(point_pairs, prior, camera, detections, box, config)
    edges = edge_gate(segments, detections.segments, edge_ids, config, landmarks)
    edge_pairs = match(edges.cost, config.edge_distance_px + 0.22 * config.edge_angle_deg + 5.01, assignment)
    edge_pairs = add_fragments(edge_pairs, edges, config)
    edge_pairs[:, 0] = edge_ids[edge_pairs[:, 0]]
    return point_pairs, edge_pairs


def verify_pairs(pose, camera, detections, box, pairs, config):
    points, edges = pairs
    point_ids, point_pixels, edge_ids, segments = visible(box, pose, camera)
    row_by_point = {feature: row for row, feature in enumerate(point_ids)}
    row_by_edge = {feature: row for row, feature in enumerate(edge_ids)}
    points = np.asarray([(feature, raw) for feature, raw in points if feature in row_by_point
                         and np.linalg.norm(point_pixels[row_by_point[feature]] - detections.points[raw])
                         <= 0.72 * config.point_gate_px], dtype=int).reshape(-1, 2)
    strong = topology_points(points, pose, camera, detections, box, config)
    gate = edge_gate(segments, detections.segments, edge_ids, config, strong, verification=True)
    edges = np.asarray([(feature, raw) for feature, raw in edges if feature in row_by_edge
                        and np.isfinite(gate.cost[row_by_edge[feature], raw])], dtype=int).reshape(-1, 2)
    return points, edges


def bundle_fragments(detections, pairs):
    """One total-least-squares supporting line per view and physical model edge."""
    model, raw = pairs.T
    model_ids = np.unique(model)
    combined = []
    for feature in model_ids:
        observations = detections.segments[raw[model == feature]]
        if len(observations) == 1:
            combined.append(observations[0])
            continue
        points = observations.reshape(-1, 2)
        origin = points.mean(axis=0)
        _, _, axes = np.linalg.svd(points - origin, full_matrices=False)
        distance = (points - origin) @ axes[0]
        combined.append(origin + np.array([distance.min(), distance.max()])[:, None] * axes[0])
    merged = Detections(detections.points, np.asarray(combined, dtype=float).reshape(-1, 2, 2))
    return merged, np.column_stack((model_ids, np.arange(len(model_ids))))


def view_problems(cameras, detections, box, pairs):
    result = []
    for camera, measured, (points, edges) in zip(cameras, detections, pairs, strict=True):
        merged, physical_edges = bundle_fragments(measured, edges)
        result.append(view_match(camera, merged, points, physical_edges))
    return tuple(result)


def normalized_jacobian(jacobian, pose, box):
    """Convert the Phase II left-world twist to dimensionless object-centred pose increments."""
    translation = jacobian[:, 3:]
    rotation = jacobian[:, :3] + translation @ skew(pose.translation)
    return np.concatenate((translation * np.linalg.norm(box.size_m), rotation), axis=1)


def final_observability(pose, box, matches, pairs, config, problem=None):
    if problem is None:
        problem = build_problem(box, matches)
    residual, jacobian = problem.evaluate(pose, jacobian=True)
    normalized = normalized_jacobian(jacobian, pose, box)
    weight = np.minimum(1., 3. / np.maximum(np.abs(residual), 1e-8))
    information = normalized.T @ (weight[:, None] * normalized) / 160.
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    eigenvalues = np.maximum(eigenvalues, 0.)
    tolerance = max(1e-8, eigenvalues[-1] * 1e-9)
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    min_value = float(eigenvalues[0])
    condition = float(eigenvalues[-1] / min_value) if rank == 6 else np.inf
    directions = np.array([np.bincount(EDGE_AXES[np.unique(edge[:, 0])], minlength=3) for _, edge in pairs])
    counts = np.array([[len(points), len(np.unique(edge[:, 0])), len(edge)] for points, edge in pairs])
    rms = float(np.sqrt(np.mean(residual**2))) if len(residual) else np.inf
    support = []
    start = 0
    for observation in matches:
        count = len(observation.vertex_ids)
        local = residual[start:start + count].reshape(-1, 2)
        support.append(len(local) >= config.min_features_per_view and
                       np.median(np.linalg.norm(local, axis=1)) <= config.support_median_px)
        start += count
    return (Observability(rank, min_value, condition, eigenvalues, information, eigenvectors[:, 0], np.diag(information),
                         directions, counts, rms), np.asarray(support, dtype=bool))


def score(pose, box, cameras, detections, pairs, config):
    """A common score with one raw-observation denominator across all candidates."""
    raw_matches = [view_match(camera, measured, points, edges) for camera, measured, (points, edges) in
                   zip(cameras, detections, pairs, strict=True)]
    raw_residual = build_problem(box, raw_matches).evaluate(pose)
    assigned = sum(len(points) + len(edges) for points, edges in pairs)
    total = sum(len(view.points) + len(view.segments) for view in detections)
    return float((huber(raw_residual) + config.unmatched_penalty * (total - assigned)) / max(total, 1))


def signature(pairs):
    return frozenset((view, feature_type, int(model), int(raw)) for view, kinds in enumerate(pairs)
                     for feature_type, group in enumerate(kinds) for model, raw in group)


def pose_delta(first, second):
    mm = 1000 * np.linalg.norm(first.translation - second.translation)
    cosine = (np.trace(first.rotation @ second.rotation.T) - 1) / 2
    return float(mm), float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def refine_frame(prior, cameras, detections, box, *, method="alternating", assignment="greedy",
                 last_valid=None, config=None):
    """Fit/refine one synchronized frame without correspondence labels or reference poses."""
    if method not in ("baseline", "removal", "reassignment", "alternating"):
        raise ValueError(f"Unknown method: {method}")
    config = Config() if config is None else config
    if not len(cameras) or len(cameras) != len(detections):
        raise ValueError("Supply a nonempty camera tuple with one detections object per camera")
    if config.edge_mode not in ("legacy", "gated") or config.budget_ms <= 0 or config.max_iterations < 0:
        raise ValueError("Invalid edge mode, budget, or iteration cap")
    timings = {name: 0. for name in ("association", "verification", "fit", "observability", "total")}
    start = perf_counter()

    def elapsed():
        return (perf_counter() - start) * 1000

    def measured(stage, action):
        stage_start = perf_counter()
        result = action()
        timings[stage] += (perf_counter() - stage_start) * 1000
        return result

    def associate_all(pose):
        return tuple(associate(pose, camera, raw, box, config=config, assignment=assignment)
                     for camera, raw in zip(cameras, detections, strict=True))

    def evaluate(pose, pairs):
        def fit_problem():
            views = view_problems(cameras, detections, box, pairs)
            problem = build_problem(box, views)
            return views, problem, solve_pose(pose, problem)

        views, problem, fit = measured("fit", fit_problem)
        diagnostics, support = measured("observability", lambda: final_observability(fit.pose, box, views, pairs, config, problem))
        def verify_and_score():
            verified = tuple(verify_pairs(fit.pose, camera, raw, box, pair, config)
                             for camera, raw, pair in zip(cameras, detections, pairs, strict=True))
            return verified, score(fit.pose, box, cameras, detections, pairs, config)

        verified, objective = measured("verification", verify_and_score)
        detail = "valid"
        if not fit.valid or diagnostics.rank < 6 or diagnostics.min_eigenvalue < config.min_eigenvalue or diagnostics.condition > config.max_condition:
            detail = "unobservable"
        elif np.count_nonzero(support) < config.min_views:
            detail = "insufficient_support"
        elif diagnostics.residual_rms_px > config.max_residual_rms_px or not np.isfinite(objective):
            detail = "high_residual"
        return Candidate(fit.pose, pairs, verified, diagnostics, support, objective, detail == "valid", detail)

    initial = measured("association", lambda: associate_all(prior))
    current = evaluate(prior, initial)
    attempted = current
    known = signature(initial)
    visited = {known}
    minimum = config.min_retention * len(known)
    ended = "baseline" if method == "baseline" else "iteration_limit"
    initial_time = elapsed()
    on_time = initial_time < config.budget_ms
    best = current if current.valid and on_time else None
    history = [Iteration(current.objective, 0., *pose_delta(current.pose, prior), initial_time,
                         best is not None, current.valid, sum(len(x) for x, _ in initial),
                         sum(len(x) for _, x in initial), current.reason if on_time else "budget_exhausted")]
    limit = 0 if method == "baseline" else config.max_iterations if method == "alternating" else min(1, config.max_iterations)
    for _ in range(limit):
        if elapsed() >= config.budget_ms:
            ended = "budget_exhausted"
            break
        if method == "removal":
            proposed = current.verified_pairs
        else:
            proposed = measured("association", lambda: associate_all(current.pose))
        next_signature = signature(proposed)
        change = len(known ^ next_signature) / max(len(known | next_signature), 1)
        if next_signature in visited and next_signature != known:
            ended = "cycle"
            break
        if elapsed() >= config.budget_ms:
            ended = "budget_exhausted"
            break
        attempted = evaluate(current.pose, proposed)
        mm, deg = pose_delta(attempted.pose, current.pose)
        enough = len(next_signature) >= minimum
        objective_change = attempted.objective - current.objective
        objective_ok = objective_change <= config.objective_tolerance or not current.valid
        within_budget = elapsed() < config.budget_ms
        accepted = attempted.valid and enough and objective_ok and within_budget
        detail = ("budget_exhausted" if not within_budget else "over_pruned" if not enough else
                  attempted.reason if not attempted.valid else "objective_regressed" if not objective_ok else "valid")
        history.append(Iteration(attempted.objective, change, mm, deg, elapsed(), accepted, attempted.valid,
                                sum(len(x) for x, _ in proposed), sum(len(x) for _, x in proposed), detail))
        if not accepted:
            ended = detail
            break
        best = current = attempted
        visited.add(next_signature)
        if (next_signature == known and mm <= config.step_mm and deg <= config.step_deg
                and abs(objective_change) <= config.objective_tolerance):
            ended = "converged"
            break
        known = next_signature
    if best is None:
        ended = "budget_exhausted" if elapsed() >= config.budget_ms else attempted.reason
        result = attempted
        pose = prior if last_valid is None else last_valid
    else:
        result = best
        pose = best.pose
    timings["total"] = elapsed()
    return RefinementResult(pose, best is not None, ended, result.diagnostics, tuple(x for x, _ in result.pairs),
                            tuple(x for _, x in result.pairs), tuple(x for x, _ in result.verified_pairs),
                            tuple(x for _, x in result.verified_pairs), result.support, tuple(history), timings)
