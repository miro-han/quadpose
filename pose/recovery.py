"""Causal pose recovery with bounded image-only PnP and explicit output status."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from time import perf_counter

import cv2
import numpy as np

from pose.fusion import SIGNS, LocalFit, Pose, build_problem, edge_pairs, rotation_vector

from .hypotheses import (
    Selection, association, evaluate_fit, evaluate_hypotheses, pose_distance, select_pose, support_by_camera,
    to_selection,
)


MODES = ('hold', 'adaptive', 'velocity', 'multi', 'pnp', 'partial', 'full')


@dataclass(frozen=True)
class RecoveryConfig:
    prediction_seconds: float = .25
    max_prediction_frames: int = 5
    max_search_scale: float = 3.2
    max_probe_seeds: int = 6
    pnp_quads_per_view: int = 3
    pnp_finalists: int = 4
    min_partial_eigenvalue: float = 1.
    min_partial_corner_count: int = 4


def predict_pose(history, timestamp):
    """Extrapolate two measured samples with world translation and left SO(3) increments."""
    last_time, last_pose = history[-1]
    if len(history) < 2:
        return last_pose
    previous_time, previous_pose = history[-2]
    ratio = (timestamp - last_time) / (last_time - previous_time)
    if ratio <= 0:
        return last_pose
    ratio = min(ratio, 4.)
    delta = last_pose.translation - previous_pose.translation
    rotation = cv2.Rodrigues(last_pose.rotation @ previous_pose.rotation.T)[0].ravel()
    return Pose(rotation_vector(rotation * ratio) @ last_pose.rotation, last_pose.translation + delta * ratio)


def probe_seeds(prior, count):
    """Alternating local image-association basins in world axes."""
    for i in range(count):
        axis = np.zeros(3)
        axis[(i // 2) % 3] = -1. if i % 2 else 1.
        yield Pose(rotation_vector(axis * np.deg2rad(16.)) @ prior.rotation,
                   prior.translation + .012 * axis)


def ordered_polygon(points):
    center = points.mean(axis=0)
    offsets = points - center
    return points[np.argsort(np.arctan2(offsets[:, 1], offsets[:, 0]))]


def face_vertices(box):
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for sign in (-1, 1):
            vertices = box.vertices[SIGNS[:, axis] == sign]
            flat = vertices[:, other]
            order = np.argsort(np.arctan2(flat[:, 1], flat[:, 0]))
            yield vertices[order]


def image_quads(detection, maximum):
    """Rank bounded image-only quadrilaterals by boundary segment support."""
    points = detection.points[np.isfinite(detection.points).all(axis=1)]
    if len(points) < 4:
        return []
    if len(points) > 9:
        if len(detection.segments):
            ends = detection.segments.reshape(-1, 2)
            distances = np.linalg.norm(points[:, None] - ends[None], axis=2).min(axis=1)
            points = points[np.argsort(distances, kind='stable')[:9]]
        else:
            indices = cv2.convexHull(points.astype(np.float32), returnPoints=False)[:, 0]
            others = np.setdiff1d(np.arange(len(points)), indices)
            points = points[np.r_[indices, others][:9]]
    candidates = []
    for subset in combinations(points, 4):
        quad = ordered_polygon(np.array(subset))
        polygon = quad.astype(np.float32)
        area = cv2.contourArea(polygon)
        if area < 150 or not cv2.isContourConvex(polygon):
            continue
        boundary = np.stack((quad, np.roll(quad, -1, axis=0)), axis=1)
        if np.min(np.linalg.norm(boundary[:, 1] - boundary[:, 0], axis=1)) < 9:
            continue
        count = len(edge_pairs(boundary, detection.segments, 13.))
        candidates.append((-count, -area, quad))
    candidates.sort(key=lambda item: item[:2])
    return [item[2] for item in candidates[:maximum]]


def pnp_proposals(cameras, detections, box, config):
    """Lift image quadrilaterals without using ground-truth poses or feature IDs."""
    faces = list(face_vertices(box))
    for camera, detection in zip(cameras, detections, strict=True):
        for quad in image_quads(detection, config.pnp_quads_per_view):
            for face in faces:
                for oriented in (quad, quad[::-1]):
                    for shift in range(4):
                        pixels = np.roll(oriented, shift, axis=0).copy()
                        try:
                            count, rotations, translations, errors = cv2.solvePnPGeneric(
                                face, pixels, camera.intrinsic, None, flags=cv2.SOLVEPNP_IPPE)
                        except cv2.error:
                            continue
                        for i in range(count):
                            if not np.isfinite(errors[i]).all() or float(errors[i, 0]) > 5.:
                                continue
                            local_rotation = cv2.Rodrigues(rotations[i])[0]
                            local_translation = translations[i].ravel()
                            depths = (box.vertices @ local_rotation.T + local_translation)[:, 2]
                            if not np.isfinite(depths).all() or np.min(depths) <= .04:
                                continue
                            yield Pose(camera.rotation.T @ local_rotation,
                                       camera.center + camera.rotation.T @ local_translation)


def pnp_finalists(prior, cameras, detections, box, config):
    finalists = []
    generated = 0
    for candidate in pnp_proposals(cameras, detections, box, config):
        generated += 1
        claims = tuple(association(candidate, cam, detected, box, 1.)
                       for cam, detected in zip(cameras, detections, strict=True))
        problems = tuple(build_problem(box, [claim]) for claim in claims)
        _, residuals, inliers, supported = support_by_camera(LocalFit(candidate, True, 0., 0.), problems)
        if supported.sum() < 2:
            continue
        distance, rotation = pose_distance(prior, candidate)
        quality = (-int(supported.sum()), -int(inliers[supported].sum()),
                   float(residuals[supported].mean()) + min(2., .018 * rotation + 5 * distance))
        if any(pose_distance(candidate, existing)[0] < .005 and pose_distance(candidate, existing)[1] < 2.
               for _, existing in finalists):
            continue
        finalists.append((quality, candidate))
        finalists.sort(key=lambda item: item[0])
        del finalists[config.pnp_finalists:]
    return [pose for _, pose in finalists], generated


def partial_pose(prior, cameras, detections, box, scale, config):
    matches = tuple(association(prior, camera, data, box, scale)
                    for camera, data in zip(cameras, detections, strict=True))
    problems = tuple(build_problem(box, [match]) for match in matches)
    active = [i for i, match in enumerate(matches) if match.corners + match.edges >= 3]
    if not 1 <= len(active) <= 2:
        return None
    row = evaluate_fit(prior, cameras, detections, box, matches, problems, active, scale=scale)
    support = row.supported & np.isin(np.arange(len(cameras)), active)
    stable = len(active) > 1 or matches[active[0]].corners >= config.min_partial_corner_count
    accepted = (stable and row.fit.valid and row.rank == 6 and row.min_eigenvalue >= config.min_partial_eigenvalue
                and row.condition < 1000 and np.all(support[active]) and row.fit.rms <= 5.5)
    reason = 'partial_camera_measurement' if accepted else 'partial_geometry_rejected'
    return to_selection(row.fit.pose, accepted, reason, support, matches,
                        {'rank': row.rank, 'condition': row.condition, 'min_eigenvalue': row.min_eigenvalue,
                         'fit_views': tuple(active), 'residual_rms_px': row.fit.rms, 'iterations': 0})


class RecoveryTracker:
    """Maintain measured history independently of predicted display poses."""

    def __init__(self, box, initial_pose=None, *, policy='cross', mode='full', config=None):
        if mode not in MODES:
            raise ValueError(f'Unknown recovery mode: {mode}')
        self.box, self.initial_pose, self.policy = box, initial_pose, policy
        self.mode, self.config = mode, config or RecoveryConfig()
        self.history = []
        self.misses = 0
        self.last_timestamp = None

    def _cross(self, prior, cameras, detections, scale):
        matrix = evaluate_hypotheses(prior, cameras, detections, self.box, search_scale=scale)
        return select_pose(matrix, self.policy)

    def _accepted(self, choice, prior, *, reinitializing=False):
        if choice is None or not choice.valid or choice.diagnostics.get('rank', 0) != 6:
            return False
        if choice.diagnostics.get('residual_rms_px', np.inf) > 7.:
            return False
        shift, angle = pose_distance(choice.pose, prior)
        return reinitializing or (shift <= .07 + .01 * self.misses and angle <= 34. + 4 * self.misses)

    def step(self, timestamp_s, cameras, detections):
        started = perf_counter()
        cameras, detections = tuple(cameras), tuple(detections)
        if len(cameras) != 4 or len(detections) != 4 or not np.isfinite(timestamp_s):
            raise ValueError('Use a finite timestamp and four aligned camera slots')
        if self.last_timestamp is not None and timestamp_s <= self.last_timestamp:
            raise ValueError('Recovery timestamps must strictly increase')
        self.last_timestamp = timestamp_s
        prior = self.history[-1][1] if self.history else self.initial_pose
        elapsed = timestamp_s - self.history[-1][0] if self.history else np.inf
        prediction_allowed = bool(self.history and elapsed <= self.config.prediction_seconds
                                  and self.misses < self.config.max_prediction_frames)
        moving = self.mode not in ('hold', 'adaptive')
        if moving and prediction_allowed:
            prior = predict_pose(self.history, timestamp_s)
        scale = (min(self.config.max_search_scale, 1. + .55 * self.misses)
                 if self.mode not in ('hold', 'velocity') else 1.)
        attempts, generated, choice, accepted = [], 0, None, False
        available = sum(len(data.points) + len(data.segments) >= 3 for data in detections)
        if prior is not None and available:
            choice = self._cross(prior, cameras, detections, scale)
            attempts.append('cross')
            accepted = self._accepted(choice, prior)
            if not accepted and self.mode in ('partial', 'full'):
                partial = partial_pose(prior, cameras, detections, self.box, scale, self.config)
                attempts.append('partial')
                if partial is not None:
                    choice = partial
                    accepted = self._accepted(choice, prior)
            if not accepted and self.misses and available >= 3 and self.mode in ('multi', 'full'):
                for seed in probe_seeds(prior, self.config.max_probe_seeds):
                    probe = self._cross(seed, cameras, detections, scale)
                    attempts.append('multi')
                    if self._accepted(probe, prior):
                        choice, accepted = probe, True
                        break
        available_corners = sum(len(data.points) >= 4 for data in detections)
        if not accepted and (self.misses or prior is None) and self.mode in ('pnp', 'full') and available_corners >= 2:
            reference = prior if prior is not None else Pose(np.eye(3), np.zeros(3))
            candidates, generated = pnp_finalists(reference, cameras, detections, self.box, self.config)
            attempts.append('pnp')
            for seed in candidates:
                probe = self._cross(seed, cameras, detections, 1.)
                if not probe.valid and self.mode == 'full':
                    probe = partial_pose(seed, cameras, detections, self.box, 1., self.config)
                if self._accepted(probe, reference, reinitializing=True):
                    choice, accepted = replace(probe, reason='image_pnp_reinitialized'), True
                    break

        diagnostics = {**(choice.diagnostics if choice else {}), 'search_scale': scale,
                       'recovery_attempts': tuple(attempts), 'pnp_proposals': generated,
                       'prior_source': ('constant_velocity' if moving and prediction_allowed else 'last_measured_or_seed'),
                       'recovery_elapsed_ms': (perf_counter() - started) * 1000}
        if accepted:
            reacquired = self.misses > 0 or not self.history and self.initial_pose is None
            if reacquired and not prediction_allowed:
                self.history.clear()  # A discontinuous reinitialization cannot seed old velocity.
            self.history = (self.history + [(timestamp_s, choice.pose)])[-2:]
            self.misses = 0
            return replace(choice, diagnostics=diagnostics, reacquired=reacquired)

        self.misses += 1
        empty = tuple(np.empty((0, 2), int) for _ in cameras)
        status = 'predicted' if prediction_allowed else 'lost'
        reason = choice.reason if choice and not choice.valid else 'measurement_rejected'
        return Selection(prior if prediction_allowed else None, False, status, reason,
                         np.zeros(4, bool), empty, empty, diagnostics)
