"""Compare every diagonal position using shared geometric visibility and pose information."""

import argparse
import csv
from itertools import combinations, product
import json
import os
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

HALF_RIG = .65
HALF_BOX = np.array([.190, .120, .085]) / 2
LENGTH = np.linalg.norm(2 * HALF_BOX)
IMAGE_SIZE = (960, 720)
FOCAL = 900.
AXES = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
CAMERA_NAMES = ('front', 'side', 'top', 'diagonal')
SIGNS = np.array(list(product((-1, 1), repeat=3)))
CORNERS = SIGNS * HALF_BOX
EDGES = np.array([(a, b) for a, b in combinations(range(8), 2) if sum(SIGNS[a] != SIGNS[b]) == 1])
SAMPLES = 21
FRACTIONS = (np.arange(SAMPLES) + .5) / SAMPLES
EDGE_POINTS = CORNERS[EDGES[:, :1]] + FRACTIONS[None, :, None] * np.diff(CORNERS[EDGES], axis=1)
POINTS = np.concatenate((CORNERS, EDGE_POINTS.reshape(-1, 3)))
FACES = np.isclose(np.abs(POINTS), HALF_BOX)
HAND_RADIUS = .085
FLOOR = -.65
CLEARANCE = .08
POLE = np.array([.56, -.56])
POLE_RADIUS = .022
POLE_Z = (FLOOR, .68)
CASES = ('ideal', 'floor_clearance', 'floor_and_pole')
NORMS = 4 * (2 * len(CORNERS) + len(EDGES))


def octant(signs):
    return ''.join(f'{axis}{"+" if sign > 0 else "-"}' for axis, sign in zip('xyz', signs, strict=True))


HANDS = {'none': None, 'front': [0, -1, 0], 'back': [0, 1, 0],
         'side': [1, 0, 0], 'other_side': [-1, 0, 0], 'above': [0, 0, 1], 'below': [0, 0, -1]}
HANDS.update({octant(sign): sign for sign in SIGNS})
HAND_DIRECTIONS = np.array([direction / np.linalg.norm(direction) for direction in list(HANDS.values())[1:]])


def camera_centers(signs, mode):
    diagonal = np.asarray(signs) * HALF_RIG / (np.sqrt(3) if mode == 'equal_radius' else 1.)
    return np.vstack((AXES * HALF_RIG, diagonal))


def camera_orientations(centers):
    forward = -centers / np.linalg.norm(centers, axis=1, keepdims=True)
    up = np.tile([0., 0., 1.], (len(centers), 1))
    up[np.abs(forward[:, 2]) > .999] = [0., 1., 0.]
    right = np.cross(forward, up)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    return np.stack((right, np.cross(forward, right), forward), axis=1)


def euler(angles):
    sx, sy, sz = np.sin(angles)
    cx, cy, cz = np.cos(angles)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @
            np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @
            np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def poses(frames):
    for phase in np.linspace(0, 2 * np.pi, frames, endpoint=False):
        a, b, c = np.sin([phase, 2 * phase, 3 * phase])
        yield euler(np.radians([28 * a, 38 * b, 40 * c])), .047 * np.array([a, b, c])
    for phase in np.linspace(0, 2 * np.pi, frames, endpoint=False):
        a, b, c = np.sin([phase + .25, 1.5 * phase, phase + np.pi / 2])
        yield euler(np.radians([32 * b, 27 * a + 8, 60 * c])), np.array([.083 * a, .067 * c, .035 * b + .04])
    for phase in np.linspace(0, 2 * np.pi, frames, endpoint=False):
        a, b, c = np.sin([2 * phase + .49, phase + 1.2, 3 * phase])
        yield euler(np.radians([40 * a - 12, 43 * b, 68 * c])), np.array([.06 * b, .057 * c, .04 * a - .045])


def cuboid_visible(centers, rotation, translation):
    local = (centers - translation) @ rotation
    front = local[:, None] * np.sign(POINTS[None]) > HALF_BOX + 1e-10
    return np.any(front & FACES, axis=-1)


def hand_visibility(centers, world, rotation, translation, gap):
    support = np.abs(HAND_DIRECTIONS @ rotation) @ HALF_BOX
    hands = translation + (support + HAND_RADIUS + gap)[:, None] * HAND_DIRECTIONS
    rays = world[None] - centers[:, None]
    hand_vectors = hands[:, None, None] - centers[None, :, None]
    parameter = np.sum(hand_vectors * rays, axis=-1) / np.sum(rays * rays, axis=-1)
    closest = np.clip(parameter, 0, 1)[..., None] * rays
    visible = np.sum((hand_vectors - closest)**2, axis=-1) > HAND_RADIUS**2
    return np.concatenate((np.ones((1, *visible.shape[1:]), dtype=bool), visible))


def pole_blocks(centers, world):
    """Segment against a finite solid vertical cylinder, including a vertical ray."""
    rays = world[None] - centers[:, None]
    origin_xy = centers[:, None, :2] - POLE
    a = np.sum(rays[..., :2] ** 2, axis=-1)
    b = np.sum(origin_xy * rays[..., :2], axis=-1)
    c = np.sum(origin_xy ** 2, axis=-1) - POLE_RADIUS**2
    discriminant = b * b - a * c
    root = np.sqrt(np.maximum(discriminant, 0))
    radial_moves = a > 1e-14
    safe_a = np.where(radial_moves, a, 1.)
    radial_low = np.where(radial_moves, (-b - root) / safe_a, -np.inf)
    radial_high = np.where(radial_moves, (-b + root) / safe_a, np.inf)
    can_hit_disk = np.where(radial_moves, discriminant >= 0, c <= 0)
    z = rays[..., 2]
    vertical_moves = np.abs(z) > 1e-14
    safe_z = np.where(vertical_moves, z, 1.)
    first, last = [(limit - centers[:, None, 2]) / safe_z for limit in POLE_Z]
    vertical_low = np.where(vertical_moves, np.minimum(first, last), -np.inf)
    vertical_high = np.where(vertical_moves, np.maximum(first, last), np.inf)
    can_hit_height = vertical_moves | ((POLE_Z[0] <= centers[:, None, 2]) & (centers[:, None, 2] <= POLE_Z[1]))
    return (can_hit_disk & can_hit_height & (np.maximum(np.maximum(radial_low, vertical_low), 0.) <=
            np.minimum(np.minimum(radial_high, vertical_high), 1.)))


def project_and_info(centers, camera_r, world, translation):
    viewed = np.einsum('vij,vpj->vpi', camera_r, world[None] - centers[:, None])
    depth = viewed[..., 2]
    pixel = FOCAL * viewed[..., :2] / depth[..., None] + np.asarray(IMAGE_SIZE) / 2
    in_frame = (depth > .02) & np.all((pixel >= 0) & (pixel < IMAGE_SIZE), axis=-1)
    factors = FOCAL / depth
    projection_derivative = np.zeros((*depth.shape, 2, 3))
    projection_derivative[..., 0, 0] = projection_derivative[..., 1, 1] = factors
    projection_derivative[..., :, 2] = -viewed[..., :2] / depth[..., None] * factors[..., None]
    local_world = world - translation
    relative_rotation = np.cross(np.eye(3)[None], local_world[:, None]).transpose(0, 2, 1)
    pose_jacobian = np.concatenate((np.broadcast_to(LENGTH * np.eye(3), relative_rotation.shape), relative_rotation), axis=-1)
    image_jacobian = (projection_derivative @ camera_r[:, None]) @ pose_jacobian
    corner_jac = image_jacobian[:, :8]
    corner_info = np.einsum('vpai,vpaj->vpij', corner_jac, corner_jac)
    edge_direction = pixel[:, EDGES[:, 1]] - pixel[:, EDGES[:, 0]]
    normal = np.stack((-edge_direction[..., 1], edge_direction[..., 0]), axis=-1)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(norm, 1e-10)
    edge_jac = np.einsum('vea,vesai->vesi', normal, image_jacobian[:, 8:].reshape(len(centers), 12, SAMPLES, 2, 6))
    edge_info = np.einsum('vesi,vesj->vesij', edge_jac, edge_jac).reshape(len(centers), -1, 6, 6) / SAMPLES
    return in_frame, np.concatenate((corner_info, edge_info), axis=1) / NORMS


def statistics(visible, contributions):
    all_support = visible.sum(axis=1)
    axis_support = visible[:, :3].sum(axis=1)
    losses = (axis_support[0] >= 2) & (axis_support < 2)
    recoveries = losses & (all_support >= 2)
    camera_fisher = np.einsum('hvp,vpij->hvij', visible.astype(float), contributions)
    full = camera_fisher.sum(axis=1)
    full_and_drops = np.concatenate((full[:, None], full[:, None] - camera_fisher), axis=1)
    spectrum = np.maximum(np.linalg.eigvalsh(full_and_drops), 0.)
    full_rank = spectrum[..., 0] > np.maximum(1e-8, 1e-9 * spectrum[..., -1])
    condition = np.divide(spectrum[..., -1], spectrum[..., 0], out=np.full(full_rank.shape, np.inf), where=full_rank)
    camera_support = visible.any(axis=-1)
    camera_count = camera_support.sum(axis=-1)
    drop_support = all_support[:, None] - visible
    drop_corner = np.mean(drop_support[..., :8] >= 2, axis=-1)
    drop_edge = np.mean(drop_support[..., 8:] >= 2, axis=-1)
    return {'all_support': all_support, 'axis_support': axis_support, 'lost': losses, 'recovered': recoveries,
            'spectrum': spectrum, 'full_rank': full_rank, 'condition': condition,
            'drop_corner_2plus': drop_corner, 'drop_edge_2plus': drop_edge,
            'camera_counts': np.concatenate((camera_count[:, None], camera_count[:, None] - camera_support), axis=1)}


def quantile(values, probability):
    """Linear quantile with finite values plus explicit infinite-condition handling."""
    ordered = np.sort(np.asarray(values).ravel())
    index = probability * (len(ordered) - 1)
    low, high = ordered[int(np.floor(index))], ordered[int(np.ceil(index))]
    return float(low if low == high else high if np.isinf(high) else low + (index % 1) * (high - low))


def report_rows(layout, mode, scenario, entries):
    stacks = {key: np.stack([row[key] for row in entries]) for key in entries[0]}
    rows, dropped_rows = [], []
    groups = [(name, [index]) for index, name in enumerate(HANDS)]
    groups.append(('all_hands', list(range(1, len(HANDS)))))
    for name, selected in groups:
        data = {key: values[:, selected] for key, values in stacks.items()}
        keys = dict(distance_mode=mode, octant=layout, scenario=scenario, hand=name, pose_hand_count=len(entries) * len(selected))
        summary = dict(keys)
        for kind, feature in (('corner', slice(0, 8)), ('edge_sample', slice(8, None))):
            all_views, axes = [data[key][..., feature] for key in ('all_support', 'axis_support')]
            for threshold in (1, 2):
                visible = np.mean(all_views >= threshold)
                summary[f'{kind}_{threshold}plus_pct'] = float(100 * visible)
                summary[f'{kind}_gain_over_axes_{threshold}plus_pp'] = float(100 * (visible - np.mean(axes >= threshold)))
            lost, recovered = [int(np.count_nonzero(data[key][..., feature])) for key in ('lost', 'recovered')]
            summary[f'{kind}_lost_axis_2plus_count'] = lost
            summary[f'{kind}_recover_axis_2plus_pct'] = 100. * recovered / lost if lost else None
        for index in range(6):
            summary[f'full_lambda_{index + 1}_median'] = quantile(data['spectrum'][..., 0, index], .5)
        summary.update(full_lambda_min_p10=quantile(data['spectrum'][..., 0, 0], .1),
                       full_condition_p90=quantile(data['condition'][..., 0], .9),
                       full_rank6_pct=100 * float(np.mean(data['full_rank'][..., 0])),
                       full_3_distinct_pct=100 * float(np.mean(data['camera_counts'][..., 0] >= 3)),
                       worst_drop_lambda_min_p10=quantile(data['spectrum'][..., 1:, 0].min(axis=-1), .1),
                       worst_drop_condition_p90=quantile(data['condition'][..., 1:].max(axis=-1), .9),
                       worst_drop_corner_2plus_pct=100 * float(np.mean(data['drop_corner_2plus'].min(axis=-1))),
                       worst_drop_edge_2plus_pct=100 * float(np.mean(data['drop_edge_2plus'].min(axis=-1))),
                       worst_drop_rank6_pct=100 * float(np.mean(data['full_rank'][..., 1:].all(axis=-1))),
                       worst_drop_3_distinct_pct=100 * float(np.mean((data['camera_counts'][..., 1:] >= 3).all(axis=-1))))
        rows.append(summary)
        for index, camera in enumerate(CAMERA_NAMES, 1):
            dropped_rows.append(dict(keys, omitted=camera,
                                     lambda_min_p10=quantile(data['spectrum'][..., index, 0], .1),
                                     condition_p90=quantile(data['condition'][..., index], .9),
                                     corner_2plus_pct=100 * float(np.mean(data['drop_corner_2plus'][..., index - 1])),
                                     edge_2plus_pct=100 * float(np.mean(data['drop_edge_2plus'][..., index - 1])),
                                     rank6_pct=100 * float(np.mean(data['full_rank'][..., index])),
                                     three_distinct_pct=100 * float(np.mean(data['camera_counts'][..., index] >= 3))))
    return rows, dropped_rows


def evaluate(frames):
    trajectories = list(poses(frames))
    summaries, drops = [], []
    for (mode, signs) in product(('equal_radius', 'cube_vertex'), SIGNS):
        centers = camera_centers(signs, mode)
        camera_r = camera_orientations(centers)
        practical = centers[:, 2] >= FLOOR + CLEARANCE
        entries = {scenario: [] for scenario in CASES}
        for i, (orientation, translation) in enumerate(trajectories):
            world = POINTS @ orientation.T + translation
            framed, contributions = project_and_info(centers, camera_r, world, translation)
            base = cuboid_visible(centers, orientation, translation) & framed
            gap = .008 + .037 * (1 + np.cos(i * 2 * np.pi / frames)) / 2
            hands = hand_visibility(centers, world, orientation, translation, gap)
            clear_pole = ~pole_blocks(centers, world)
            for scenario, mask in zip(CASES, (base, base & practical[:, None], base & practical[:, None] & clear_pole), strict=True):
                entries[scenario].append(statistics(hands & mask, contributions))
        for scenario in CASES:
            row, drop = report_rows(octant(signs), mode, scenario, entries[scenario])
            summaries.extend(row)
            drops.extend(drop)
    return summaries, drops


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--poses-per-family', type=int, default=27)
    args = parser.parse_args()
    if args.poses_per_family < 3 or args.output.exists():
        parser.error('Use at least three poses per family and a new output directory')
    rows, drops = evaluate(args.poses_per_family)
    args.output.mkdir(parents=True)
    write_csv(args.output / 'layout_hands.csv', rows)
    write_csv(args.output / 'camera_drops.csv', drops)
    config = {'axis_cameras': dict(zip(CAMERA_NAMES[:3], (AXES * HALF_RIG).tolist(), strict=True)),
              'diagonal_radius_m': {'equal_radius': HALF_RIG, 'cube_vertex': HALF_RIG * np.sqrt(3)},
              'K': [[FOCAL, 0, IMAGE_SIZE[0] / 2], [0, FOCAL, IMAGE_SIZE[1] / 2], [0, 0, 1]],
              'image_size_px': IMAGE_SIZE, 'box_m': (2 * HALF_BOX).tolist(),
              'geometry': {'floor_plane_z_m': FLOOR, 'lens_clearance_m': CLEARANCE,
                           'pole_xy_m': POLE.tolist(), 'pole_radius_m': POLE_RADIUS, 'pole_z_m': POLE_Z},
              'poses_per_family': args.poses_per_family, 'edge_samples_per_edge': SAMPLES,
              'hands': dict(zip(list(HANDS)[1:], HAND_DIRECTIONS.tolist(), strict=True)),
              'hand_radius_m': HAND_RADIUS, 'translation_parameter_scale_m': LENGTH, 'info_divisor': NORMS,
              'blas_thread_env': os.environ['OPENBLAS_NUM_THREADS']}
    (args.output / 'scenario.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(rows)} layout/hand rows and {len(drops)} all-omissions rows to {args.output}')


if __name__ == '__main__':
    main()
