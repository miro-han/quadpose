"""Model-based 6-DoF fusion of calibrated point and line detections."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


SIGNS = np.array(list(product((-1, 1), repeat=3)))
EDGES = np.array([(a, b) for a in range(8) for b in range(a + 1, 8)
                  if np.count_nonzero(SIGNS[a] != SIGNS[b]) == 1])
EDGE_AXES = np.argmax(SIGNS[EDGES[:, 0]] != SIGNS[EDGES[:, 1]], axis=1)
TOKEN_FIELDS = (
    "local_valid", "local_rms_px", "local_corner_count", "local_edge_count", "local_condition",
    "all_residual_px", "all_supported", "leave_self_out_residual_px", "leave_self_out_supported",
    *(f"residual_as_camera_h{index}_px" for index in range(4)),
    *(f"support_as_camera_h{index}" for index in range(4)),
    *(f"residual_as_hypothesis_camera{index}_px" for index in range(4)),
    *(f"support_as_hypothesis_camera{index}" for index in range(4)),
    "hypothesis_valid", "local_tx_m", "local_ty_m", "local_tz_m",
    *(f"local_r{row}{column}" for row in range(3) for column in range(3)),
)


@dataclass(frozen=True)
class Pose:
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True)
class Camera:
    name: str
    intrinsic: np.ndarray
    rotation: np.ndarray
    center: np.ndarray
    image_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class BoxModel:
    size_m: tuple[float, float, float]

    @property
    def vertices(self):
        return SIGNS * (np.asarray(self.size_m) / 2)


@dataclass(frozen=True)
class Detections:
    points: np.ndarray
    segments: np.ndarray


@dataclass(frozen=True)
class AssociationLabels:
    corner_ids: np.ndarray  # -1 denotes clutter.
    edge_ids: np.ndarray


@dataclass(frozen=True)
class ViewMatch:
    camera: Camera
    vertex_ids: np.ndarray
    pixels: np.ndarray
    normals: np.ndarray
    corners: int
    edges: int
    point_pairs: np.ndarray
    segment_pairs: np.ndarray


@dataclass(frozen=True)
class LocalFit:
    pose: Pose
    valid: bool
    condition: float
    rms: float


@dataclass(frozen=True)
class Evidence:
    prior: Pose
    box: BoxModel
    matches: tuple[ViewMatch, ...]
    local: tuple[LocalFit, ...]
    hypotheses: tuple[LocalFit, ...]
    residuals: np.ndarray  # [omitted camera, evaluated camera], in pixels
    support: np.ndarray  # same indexing, no true fault mask
    all_fit: LocalFit
    all_residuals: np.ndarray
    all_support: np.ndarray


@dataclass(frozen=True)
class Estimate:
    pose: Pose
    valid: bool
    support: np.ndarray
    excluded: tuple[int, ...]


def skew(vector):
    x, y, z = np.moveaxis(vector, -1, 0)
    zeros = np.zeros_like(x)
    return np.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), axis=-1).reshape((*x.shape, 3, 3))


def rotation_vector(vector):
    angle = np.linalg.norm(vector)
    cross = skew(vector)
    if angle < 1e-8:
        return np.eye(3) + cross + 0.5 * cross @ cross
    return np.eye(3) + np.sin(angle) / angle * cross + (1 - np.cos(angle)) / angle**2 * cross @ cross


def advance(pose, delta):
    rotation = rotation_vector(delta[:3])
    return Pose(rotation @ pose.rotation, rotation @ pose.translation + delta[3:])


def project(points, pose, camera):
    world = points @ pose.rotation.T + pose.translation
    xyz = (world - camera.center) @ camera.rotation.T
    homogeneous = xyz @ camera.intrinsic.T
    uv = homogeneous[:, :2] / np.maximum(homogeneous[:, 2, None], 1e-8)
    return uv, xyz[:, 2]


def visible(box, pose, camera):
    vertices = box.vertices
    uv, depth = project(vertices, pose, camera)
    inside = depth > 0
    if camera.image_size is not None:
        inside &= np.all((uv >= 0) & (uv < camera.image_size), axis=1)
    camera_local = pose.rotation.T @ (camera.center - pose.translation)
    front_faces = SIGNS * camera_local > np.asarray(box.size_m) / 2
    corners = np.flatnonzero(inside & np.any(front_faces, axis=1))
    edge_faces = front_faces[EDGES[:, 0]].copy()
    edge_faces[np.arange(len(EDGES)), EDGE_AXES] = False
    spans = np.linalg.norm(uv[EDGES[:, 1]] - uv[EDGES[:, 0]], axis=1)
    edges = np.flatnonzero(np.all(inside[EDGES], axis=1) & np.any(edge_faces, axis=1) & (spans >= 8))
    return corners, uv[corners], edges, uv[EDGES[edges]]


def line_geometry(segments):
    difference = segments[:, 1] - segments[:, 0]
    lengths = np.linalg.norm(difference, axis=1)
    directions = difference / np.maximum(lengths[:, None], 1e-8)
    return directions, lengths


def greedy_pairs(costs, gate):
    pairs = np.argwhere(costs < gate)
    occupied_rows, occupied_columns, selected = set(), set(), []
    for row, column in pairs[np.argsort(costs[pairs[:, 0], pairs[:, 1]], kind="stable")]:
        if row not in occupied_rows and column not in occupied_columns:
            selected.append((row, column))
            occupied_rows.add(row)
            occupied_columns.add(column)
    return np.asarray(selected, dtype=int).reshape(-1, 2)


def edge_pairs(predicted, observed, gate):
    predicted_dirs, predicted_lengths = line_geometry(predicted)
    dirs, lengths = line_geometry(observed)
    midpoint_delta = observed.mean(axis=1)[None] - predicted.mean(axis=1)[:, None]
    along = np.abs(np.einsum("ijk,ik->ij", midpoint_delta, predicted_dirs))
    normals = predicted_dirs[:, ::-1] * [-1, 1]
    across = np.abs(np.einsum("ijk,ik->ij", midpoint_delta, normals))
    angle = np.arccos(np.clip(np.abs(predicted_dirs @ dirs.T), 0, 1))
    length_ratio = lengths[None] / predicted_lengths[:, None]
    cost = across + 0.35 * along + 24 * angle
    reject = (angle > np.deg2rad(30)) | (length_ratio < 0.28) | (length_ratio > 1.4)
    reject |= (along > 0.44 * predicted_lengths[:, None]) | (lengths[None] < 6)
    cost[reject] = np.inf
    return greedy_pairs(cost, gate)


def associate_view(prior, camera, detections, box):
    corners, projected_points, edges, projected_segments = visible(box, prior, camera)
    gate = np.clip(0.029 * (camera.intrinsic[0, 0] + camera.intrinsic[1, 1]) / 2, 15, 40)
    points = greedy_pairs(np.linalg.norm(projected_points[:, None] - detections.points, axis=2), gate)
    segments = edge_pairs(projected_segments, detections.segments, gate)
    return np.column_stack((corners[points[:, 0]], points[:, 1])), np.column_stack((edges[segments[:, 0]], segments[:, 1]))


def oracle_pairs(labels):
    points = np.flatnonzero(labels.corner_ids >= 0)
    edges = np.flatnonzero(labels.edge_ids >= 0)
    return np.column_stack((labels.corner_ids[points], points)), np.column_stack((labels.edge_ids[edges], edges))


def view_match(camera, detections, point_pairs, segment_pairs):
    corner_ids, point_rows = point_pairs.T
    edge_ids, edge_rows = segment_pairs.T
    segments = detections.segments[edge_rows]
    directions, _ = line_geometry(segments)
    edge_normals = directions[:, ::-1] * [-1, 1]
    vertices = np.concatenate((np.repeat(corner_ids, 2), EDGES[edge_ids].ravel()))
    pixels = np.concatenate((np.repeat(detections.points[point_rows], 2, axis=0),
                             np.repeat(segments.mean(axis=1), 2, axis=0)))
    normals = np.concatenate((np.tile(np.eye(2), (len(corner_ids), 1)), np.repeat(edge_normals, 2, axis=0)))
    return ViewMatch(camera, vertices, pixels, normals, len(corner_ids), len(edge_ids), point_pairs, segment_pairs)


@dataclass(frozen=True)
class Problem:
    vertices: np.ndarray
    rotations: np.ndarray
    centers: np.ndarray
    intrinsics: np.ndarray
    pixels: np.ndarray
    normals: np.ndarray

    def evaluate(self, pose, jacobian=False):
        world = self.vertices @ pose.rotation.T + pose.translation
        xyz = np.einsum("nij,nj->ni", self.rotations, world - self.centers)
        homogeneous = np.einsum("nij,nj->ni", self.intrinsics, xyz)
        depths = np.maximum(homogeneous[:, 2, None], 1e-8)
        uv = homogeneous[:, :2] / depths
        residual = np.einsum("ij,ij->i", self.normals, uv - self.pixels)
        if not jacobian:
            return residual
        image_jac = (self.intrinsics[:, :2] - uv[:, :, None] * self.intrinsics[:, 2:]) / depths[:, :, None]
        motion = np.concatenate((-skew(world), np.broadcast_to(np.eye(3), (len(world), 3, 3))), axis=2)
        derivative = np.einsum("ni,nij->nj", self.normals, image_jac @ self.rotations @ motion)
        return residual, derivative


def build_problem(box, matches):
    vertices = box.vertices
    count = [len(view.vertex_ids) for view in matches]
    return Problem(
        np.concatenate([vertices[view.vertex_ids] for view in matches]),
        np.repeat(np.stack([view.camera.rotation for view in matches]), count, axis=0),
        np.repeat(np.stack([view.camera.center for view in matches]), count, axis=0),
        np.repeat(np.stack([view.camera.intrinsic for view in matches]), count, axis=0),
        np.concatenate([view.pixels for view in matches]), np.concatenate([view.normals for view in matches]),
    )


def huber(residual, cutoff=3.0):
    value = np.abs(residual)
    return np.where(value <= cutoff, value**2 / 2, cutoff * (value - cutoff / 2)).sum()


def solve_pose(prior, problem):
    if len(problem.vertices) < 6:
        return LocalFit(prior, False, np.inf, np.inf)
    residual, jac = problem.evaluate(prior, jacobian=True)
    singular = np.linalg.svd(jac, compute_uv=False)
    condition = singular[0] / max(singular[-1], 1e-8)
    if singular[-1] < 0.05 or condition > 5000:
        return LocalFit(prior, False, condition, np.inf)
    pose, damping = prior, 1e-3
    for _ in range(8):
        weights = np.minimum(1, 3 / np.maximum(np.abs(residual), 1e-8))
        normal = jac.T @ (jac * weights[:, None])
        update = np.linalg.solve(normal + damping * np.diag(np.maximum(1, np.diag(normal))), -jac.T @ (weights * residual))
        if np.linalg.norm(update) < 1e-6:
            break
        candidate = advance(pose, update)
        new_residual = problem.evaluate(candidate)
        if huber(new_residual) < huber(residual):
            pose = candidate
            residual, jac = problem.evaluate(pose, jacobian=True)
            damping = max(damping / 3, 1e-6)
        else:
            damping *= 8
    return LocalFit(pose, True, condition, float(np.sqrt(np.mean(residual**2))))


def support_metrics(fit, problems):
    residuals, supported = np.full(len(problems), np.inf), np.zeros(len(problems), dtype=bool)
    if not fit.valid:
        return residuals, supported
    for index, problem in enumerate(problems):
        count = len(problem.vertices) // 2
        if count < 3:
            continue
        feature_errors = np.linalg.norm(problem.evaluate(fit.pose).reshape(-1, 2), axis=1)
        residuals[index] = np.median(feature_errors)
        inlier_count = np.count_nonzero(feature_errors <= 7)
        supported[index] = residuals[index] <= 5.5 and inlier_count >= max(3, int(np.ceil(count * 0.52)))
    return residuals, supported


def evidence_from_matches(prior, box, matches):
    if len(matches) != 4:
        raise ValueError("Four synchronized camera slots are required. Use empty arrays in an unavailable slot.")
    problems = tuple(build_problem(box, [view]) for view in matches)
    local = tuple(solve_pose(prior, problem) for problem in problems)
    hypotheses = []
    for omit in range(4):
        included = matches[:omit] + matches[omit + 1:]
        if all(view.corners + view.edges >= 3 for view in included):
            hypotheses.append(solve_pose(prior, build_problem(box, included)))
        else:
            hypotheses.append(LocalFit(prior, False, np.inf, np.inf))
    hypotheses = tuple(hypotheses)
    held_scores = [support_metrics(fit, problems) for fit in hypotheses]
    all_fit = solve_pose(prior, build_problem(box, matches))
    all_residuals, all_support = support_metrics(all_fit, problems)
    return Evidence(prior, box, matches, local, hypotheses, np.stack([x[0] for x in held_scores]),
                    np.stack([x[1] for x in held_scores]), all_fit, all_residuals, all_support)


def analyze_frame(prior, cameras, detections, box):
    """Estimated path accepts only calibration, raw pixel detections, a model and a prior pose."""
    matches = tuple(view_match(camera, measured, *associate_view(prior, camera, measured, box))
                    for camera, measured in zip(cameras, detections, strict=True))
    return evidence_from_matches(prior, box, matches)


def analyze_oracle_frame(prior, cameras, detections, box, labels):
    """Simulation-only counterpart with ground-truth correspondence IDs, no fault labels."""
    matches = tuple(view_match(camera, measured, *oracle_pairs(ids))
                    for camera, measured, ids in zip(cameras, detections, labels, strict=True))
    return evidence_from_matches(prior, box, matches)


def normalized_rotation(rotation):
    u, _, vt = np.linalg.svd(rotation)
    u[:, -1] *= np.linalg.det(u @ vt)
    return u @ vt


def make_estimate(pose, valid, support):
    return Estimate(pose, bool(valid), support, tuple(np.flatnonzero(~support).tolist()))


def choose_result(evidence, method="reliable"):
    if method == "all":
        available = np.array([match.corners + match.edges > 0 for match in evidence.matches])
        return make_estimate(evidence.all_fit.pose, evidence.all_fit.valid, available)
    if method in ("best", "average"):
        valid_hypotheses = [index for index, fit in enumerate(evidence.hypotheses) if fit.valid]
        if not valid_hypotheses:
            return make_estimate(evidence.prior, False, np.zeros(4, dtype=bool))
        if method == "best":
            index = min(valid_hypotheses, key=lambda item: (evidence.hypotheses[item].rms, evidence.hypotheses[item].condition))
            return make_estimate(evidence.hypotheses[index].pose, True, np.arange(4) != index)
        poses = [evidence.hypotheses[index].pose for index in valid_hypotheses]
        average = Pose(normalized_rotation(np.mean([pose.rotation for pose in poses], axis=0)),
                       np.mean([pose.translation for pose in poses], axis=0))
        membership = np.any(np.arange(4)[None] != np.asarray(valid_hypotheses)[:, None], axis=0)
        return make_estimate(average, True, membership)
    if method != "reliable":
        raise ValueError(f"Unknown fusion method: {method}")

    candidates = []
    for omitted, fit in enumerate(evidence.hypotheses):
        supported = evidence.support[omitted]
        in_fit = np.arange(4) != omitted
        if fit.valid and np.all(supported[in_fit]):
            quality = float(np.mean(np.minimum(evidence.residuals[omitted, supported], 7)))
            candidates.append((-supported.sum(), quality, omitted))
    if not candidates:
        return make_estimate(evidence.prior, False, np.zeros(4, dtype=bool))
    _, _, omitted = min(candidates)
    supported = evidence.support[omitted]
    matches = tuple(match for match, keep in zip(evidence.matches, supported, strict=True) if keep)
    problems = tuple(build_problem(evidence.box, [view]) for view in evidence.matches)
    joint = solve_pose(evidence.hypotheses[omitted].pose, build_problem(evidence.box, matches))
    _, joint_support = support_metrics(joint, problems)
    final_support = supported & joint_support
    if joint.valid and np.count_nonzero(final_support) >= 3 and np.any(final_support != supported):
        matches = tuple(match for match, keep in zip(evidence.matches, final_support, strict=True) if keep)
        joint = solve_pose(evidence.hypotheses[omitted].pose, build_problem(evidence.box, matches))
        _, joint_support = support_metrics(joint, problems)
        final_support &= joint_support
    if not joint.valid or np.count_nonzero(final_support) < 3:
        return make_estimate(evidence.prior, False, final_support)
    return make_estimate(joint.pose, True, final_support)


def make_tokens(evidence):
    """One fixed finite feature vector per camera. No labels or ground truth."""
    rows = []
    for camera, (fit, match) in enumerate(zip(evidence.local, evidence.matches, strict=True)):
        rows.append([
            fit.valid, min(fit.rms, 50), match.corners, match.edges, np.log1p(min(fit.condition, 5000)),
            min(evidence.all_residuals[camera], 50), evidence.all_support[camera],
            min(evidence.residuals[camera, camera], 50), evidence.support[camera, camera],
            *np.minimum(evidence.residuals[:, camera], 50), *evidence.support[:, camera],
            *np.minimum(evidence.residuals[camera], 50), *evidence.support[camera],
            evidence.hypotheses[camera].valid, *fit.pose.translation, *fit.pose.rotation.ravel(),
        ])
    return np.asarray(rows, dtype=np.float32)
