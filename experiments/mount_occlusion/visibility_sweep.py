"""Compare geometric cuboid visibility for two camera mounts and hand approaches."""

import argparse
import csv
from itertools import combinations, product
from pathlib import Path

import numpy as np

HALF_SIZE = np.array([0.190, 0.120, 0.085]) / 2
HALF_RIG = 0.65
AXIS_CAMERAS = np.eye(3) * HALF_RIG
CAMERAS = {
    'A': np.vstack((AXIS_CAMERAS, np.full(3, HALF_RIG))),
    'B': np.vstack((AXIS_CAMERAS, np.full(3, -HALF_RIG))),
}
SIGNS = np.array(list(product((-1, 1), repeat=3)))
CORNERS = SIGNS * HALF_SIZE
EDGES = [(a, b) for a, b in combinations(range(8), 2) if np.count_nonzero(SIGNS[a] != SIGNS[b]) == 1]
EDGE_SAMPLES = 41
FRAMES = 81
RADIUS = 0.085
DIAGONAL = np.ones(3) / np.sqrt(3)
APPROACHES = {'none': None, 'positive_corner': DIAGONAL, 'negative_corner': -DIAGONAL}


def rotation(angles):
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def poses():
    for phase in np.linspace(0, 2 * np.pi, FRAMES, endpoint=False):
        angles = np.deg2rad([22 * np.sin(phase), 28 * np.sin(2 * phase), 18 * np.sin(3 * phase)])
        center = 0.045 * np.sin([phase, 2 * phase, 3 * phase])
        gap = 0.008 + 0.042 * (1 + np.cos(phase)) / 2
        yield rotation(angles), center, gap


def features():
    edge_ends = CORNERS[np.array(EDGES)]
    fractions = (np.arange(EDGE_SAMPLES) + 0.5) / EDGE_SAMPLES
    edge_samples = edge_ends[:, :1] + fractions[None, :, None] * np.diff(edge_ends, axis=1)
    points = np.concatenate((CORNERS, edge_samples.reshape(-1, 3)))
    faces = np.isclose(np.abs(points), HALF_SIZE)
    return points, faces


def visibility(camera_centers, points, faces, orient, center, hand):
    # A feature of a convex cuboid is visible if any incident face faces this camera.
    local_cameras = (camera_centers - center) @ orient
    front = local_cameras[:, None, :] * np.sign(points[None, :, :]) > HALF_SIZE + 1e-12
    visible = np.any(front & faces[None, :, :], axis=2)
    if hand is None:
        return visible
    world_points = points @ orient.T + center
    rays = world_points[None, :, :] - camera_centers[:, None, :]
    camera_to_hand = hand - camera_centers[:, None, :]
    t = np.sum(camera_to_hand * rays, axis=2) / np.sum(rays * rays, axis=2)
    closest = t.clip(0, 1)[..., None] * rays
    occluded = np.sum((camera_to_hand - closest)**2, axis=2) <= RADIUS**2
    return visible & ~occluded


def measure():
    points, faces = features()
    rows = []
    for index, (orient, center, gap) in enumerate(poses()):
        for layout, cameras in CAMERAS.items():
            clear = visibility(cameras, points, faces, orient, center, None)
            for approach, direction in APPROACHES.items():
                hand = None
                if direction is not None:
                    # The tangent separation along this approach direction keeps the sphere outside the cuboid.
                    support = np.abs(orient.T @ direction) @ HALF_SIZE
                    hand = center + (support + RADIUS + gap) * direction
                visible = visibility(cameras, points, faces, orient, center, hand)
                if np.any(visible & ~clear):
                    raise RuntimeError('An occluder created new visible points')
                views = visible.sum(axis=0)
                for threshold in (1, 2):
                    coverage = views >= threshold
                    rows.append(dict(frame=index, layout=layout, approach=approach, min_views=threshold,
                                     corners=int(coverage[:8].sum()),
                                     edge_sample_fraction=float(coverage[8:].mean())))
    return rows


def summarize(rows):
    summary = []
    for layout, approach, threshold in product(CAMERAS, APPROACHES, (1, 2)):
        group = [row for row in rows if (row['layout'], row['approach'], row['min_views']) ==
                 (layout, approach, threshold)]
        summary.append(dict(layout=layout, approach=approach, min_views=threshold, frames=len(group),
                            mean_corner_pct=100 * float(np.mean([row['corners'] for row in group])) / 8,
                            mean_edge_sample_pct=100 * float(np.mean([row['edge_sample_fraction'] for row in group]))))
    return summary


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rows = measure()
    centered = [row for row in rows if row['frame'] == 0 and row['approach'] == 'none']
    actual = [(row['corners'], round(row['edge_sample_fraction'] * 12)) for row in centered]
    if actual != [(7, 9), (7, 9), (8, 12), (7, 9)]:
        raise RuntimeError(f'Axis-aligned cuboid visibility sanity check failed: {actual}')
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'summary.csv', summarize(rows))
    write_csv(args.output / 'per_frame.csv', rows)
    print(f'Axis-aligned sanity passed; wrote {len(rows)} rows to {args.output}')


if __name__ == '__main__':
    main()
