"""Cross-view candidate evaluation with fixed image claims and bounded held-out evidence."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from pose.fusion import (
    BoxModel, Camera, Detections, LocalFit, Pose, associate_view, build_problem,
    edge_pairs, greedy_pairs, solve_pose, view_match, visible,
)


POLICIES = ('all', 'best', 'cross', 'cross_verify', 'cross_alternating')


@dataclass(frozen=True)
class Hypothesis:
    omitted: int | None
    fit_views: tuple[int, ...]
    fit: LocalFit
    residuals_px: np.ndarray
    supported: np.ndarray
    feature_counts: np.ndarray
    inlier_counts: np.ndarray
    rank: int
    min_eigenvalue: float
    condition: float
    directional_information: np.ndarray
    translation_delta_m: float
    rotation_delta_deg: float
    association_stability: np.ndarray

    @property
    def held_out_residual_px(self):
        return None if self.omitted is None else float(self.residuals_px[self.omitted])


@dataclass(frozen=True)
class HypothesisMatrix:
    prior: Pose
    cameras: tuple[Camera, ...]
    detections: tuple[Detections, ...]
    box: BoxModel
    matches: tuple
    problems: tuple
    triples: tuple[Hypothesis, ...]
    all: Hypothesis
    timings_ms: dict[str, float]


@dataclass(frozen=True)
class Selection:
    pose: Pose | None
    valid: bool
    status: str
    reason: str
    support: np.ndarray
    point_pairs: tuple[np.ndarray, ...]
    segment_pairs: tuple[np.ndarray, ...]
    diagnostics: dict
    reacquired: bool = False


def pose_distance(left, right):
    cosine = np.clip((np.trace(left.rotation.T @ right.rotation) - 1.0) / 2.0, -1, 1)
    return (float(np.linalg.norm(left.translation - right.translation)),
            float(np.degrees(np.arccos(cosine))))


def observation_geometry(box, problem, pose):
    """Scale translation by box diagonal and retain the full robust information matrix."""
    values, jacobian = problem.evaluate(pose, jacobian=True)
    jacobian[:, 3:] *= np.linalg.norm(box.size_m)
    weights = np.minimum(1., 3. / np.maximum(np.abs(values), 1e-8))
    scaled = jacobian * np.sqrt(weights[:, None])
    information = scaled.T @ scaled
    eigenvalues = np.maximum(np.linalg.eigvalsh(information), 0)
    tolerance = max(.05 ** 2, eigenvalues[-1] * 1e-8)
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    condition = float(np.sqrt(eigenvalues[-1] / eigenvalues[0])) if rank == 6 else float('inf')
    return rank, float(eigenvalues[0]), condition, np.diag(information)


def association(prior, camera, detected, box, scale):
    if scale == 1.:
        pairs = associate_view(prior, camera, detected, box)
    else:
        corners, pixels, edges, lines = visible(box, prior, camera)
        nominal = np.clip(.029 * np.mean([camera.intrinsic[0, 0], camera.intrinsic[1, 1]]), 15, 40)
        gate = min(110., nominal * scale)
        point_indices = greedy_pairs(np.linalg.norm(pixels[:, None] - detected.points, axis=2), gate)
        edge_indices = edge_pairs(lines, detected.segments, gate)
        pairs = (np.column_stack((corners[point_indices[:, 0]], point_indices[:, 1])),
                 np.column_stack((edges[edge_indices[:, 0]], edge_indices[:, 1])))
    return view_match(camera, detected, *pairs)


def stability(before, after):
    def names(match):
        return {('point', int(i), int(j)) for i, j in match.point_pairs} | {
            ('edge', int(i), int(j)) for i, j in match.segment_pairs}

    left, right = names(before), names(after)
    return len(left & right) / len(left | right) if left or right else np.nan


def support_by_camera(fit, problems):
    counts = np.array([len(problem.vertices) // 2 for problem in problems])
    errors = np.full(len(problems), np.inf)
    inliers = np.zeros(len(problems), int)
    for index, (problem, count) in enumerate(zip(problems, counts, strict=True)):
        if fit.valid and count:
            distances = np.linalg.norm(problem.evaluate(fit.pose).reshape(-1, 2), axis=1)
            errors[index] = np.median(distances)
            inliers[index] = np.count_nonzero(distances <= 7.)
    good = (errors <= 5.5) & (inliers >= np.maximum(3, np.ceil(.52 * counts)))
    return counts, errors, inliers, good


def evaluate_fit(prior, cameras, detections, box, matches, problems, indices, *, omitted=None, scale=1.):
    combined = build_problem(box, [matches[i] for i in indices])
    adequate = omitted is None or all(matches[i].corners + matches[i].edges >= 3 for i in indices)
    fit = solve_pose(prior, combined) if adequate else LocalFit(prior, False, np.inf, np.inf)
    counts, errors, inliers, good = support_by_camera(fit, problems)
    rank, eigenvalue, condition, directional = observation_geometry(box, combined, fit.pose)
    distance, angle = pose_distance(prior, fit.pose)
    stable = np.array([stability(before, association(fit.pose, camera, detection, box, scale))
                       for before, camera, detection in zip(matches, cameras, detections, strict=True)])
    return Hypothesis(omitted, tuple(indices), fit, errors, good, counts, inliers,
                      rank, eigenvalue, condition, directional, distance, angle, stable)


def evaluate_hypotheses(prior, cameras, detections, box, *, search_scale=1.):
    """Generate all four triples without letting a held-out image influence its fit."""
    cameras, detections = tuple(cameras), tuple(detections)
    if len(cameras) != 4 or len(detections) != 4 or not 1 <= search_scale <= 4:
        raise ValueError('Use four aligned slots and a search scale between 1 and 4')
    start = perf_counter()
    matches = tuple(association(prior, camera, data, box, search_scale)
                    for camera, data in zip(cameras, detections, strict=True))
    problems = tuple(build_problem(box, (match,)) for match in matches)
    fit_start = perf_counter()
    fixed = (prior, cameras, detections, box, matches, problems)
    triples = tuple(evaluate_fit(*fixed, [index for index in range(4) if index != skip],
                                 omitted=skip, scale=search_scale) for skip in range(4))
    all_views = evaluate_fit(*fixed, range(4), scale=search_scale)
    finish = perf_counter()
    return HypothesisMatrix(prior, cameras, detections, box, matches, problems, triples, all_views,
                            {'association_ms': (fit_start - start) * 1000,
                             'hypothesis_ms': (finish - fit_start) * 1000})


def to_selection(pose, valid, reason, support, matches, diagnostics):
    empty = np.empty((0, 2), dtype=int)
    support = np.asarray(support, dtype=bool) & bool(valid)
    points = tuple(match.point_pairs if valid and keep else empty
                   for match, keep in zip(matches, support, strict=True))
    segments = tuple(match.segment_pairs if valid and keep else empty
                     for match, keep in zip(matches, support, strict=True))
    return Selection(pose if valid else None, valid, 'measured' if valid else 'lost', reason,
                     support, points, segments, diagnostics)


def cross_score(row):
    inside = np.array(row.fit_views)
    if not row.fit.valid or row.rank < 6 or not np.all(row.supported[inside]):
        return np.inf
    rms = float(np.mean(row.residuals_px[inside]))
    count_reward = .12 * min(12., float(np.mean(row.inlier_counts[inside])))
    geometry_penalty = .12 * np.log1p(row.condition)
    temporal_penalty = min(.8, 7. * row.translation_delta_m + .016 * row.rotation_delta_deg)
    stability_penalty = .6 * (1. - float(np.mean(row.association_stability[inside])))
    held = row.omitted
    # An independently supporting holdout helps a little. Rejection or absence cannot veto a triple.
    confirmation = .75 * (1. - min(1., row.residuals_px[held] / 7.)) if row.supported[held] else 0.
    return float(rms - count_reward + geometry_penalty + temporal_penalty + stability_penalty - confirmation)


def select_pose(matrix, policy='cross', *, assignment='greedy'):
    """Original baselines, or cross-validation with optional fit-view refinement."""
    started = perf_counter()
    if policy not in POLICIES:
        raise ValueError(f'Unknown policy: {policy}')
    if policy == 'all':
        row = matrix.all
        support = row.feature_counts > 0
    elif policy == 'best':
        valid = [row for row in matrix.triples if row.fit.valid]
        row = min(valid, key=lambda item: (item.fit.rms, item.fit.condition)) if valid else None
        support = np.isin(np.arange(4), row.fit_views) if row else np.zeros(4, bool)
    else:
        row = min(matrix.triples, key=cross_score)
        row = row if np.isfinite(cross_score(row)) else None
        support = row.supported.copy() if row else np.zeros(4, bool)
    diagnostic = {**matrix.timings_ms, 'policy': policy, 'omitted': None if row is None else row.omitted,
                  'iterations': 0, 'refinement_ms': 0., 'refinement_status': None}
    if row is None or not row.fit.valid:
        diagnostic['selection_ms'] = (perf_counter() - started) * 1000
        return to_selection(None, False, 'no_accepted_hypothesis', support, matrix.matches, diagnostic)

    pose, chosen_matches = row.fit.pose, list(matrix.matches)
    diagnostic.update(rank=row.rank, condition=row.condition, min_eigenvalue=row.min_eigenvalue,
                      residual_rms_px=row.fit.rms, held_out_residual_px=row.held_out_residual_px,
                      cross_score=cross_score(row) if row.omitted is not None else None,
                      fit_views=row.fit_views, held_out_supported=(None if row.omitted is None
                                                                  else bool(row.supported[row.omitted])))
    if policy in ('cross_verify', 'cross_alternating'):
        from .refinement import refine_frame

        method = 'removal' if policy == 'cross_verify' else 'alternating'
        begin = perf_counter()
        result = refine_frame(pose, [matrix.cameras[i] for i in row.fit_views],
                              [matrix.detections[i] for i in row.fit_views], matrix.box,
                              method=method, assignment=assignment)
        diagnostic.update(refinement_ms=(perf_counter() - begin) * 1000,
                          iterations=len(result.history), refinement_status=result.reason)
        if result.valid:
            pose = result.pose
            for refined, absolute in enumerate(row.fit_views):
                chosen_matches[absolute] = view_match(matrix.cameras[absolute], matrix.detections[absolute],
                                                      result.point_pairs[refined], result.edge_pairs[refined])
                support[absolute] = result.support[refined]
            _, errors, _, independently_supported = support_by_camera(LocalFit(pose, True, 0., 0.), matrix.problems)
            support[row.omitted] = independently_supported[row.omitted]
            diagnostic.update(rank=result.diagnostics.rank, condition=result.diagnostics.condition,
                              min_eigenvalue=result.diagnostics.min_eigenvalue,
                              residual_rms_px=result.diagnostics.residual_rms_px,
                              held_out_residual_px=float(errors[row.omitted]),
                              held_out_supported=bool(support[row.omitted]))
    diagnostic['selection_ms'] = (perf_counter() - started) * 1000
    return to_selection(pose, True, 'hypothesis_accepted', support, chosen_matches, diagnostic)
