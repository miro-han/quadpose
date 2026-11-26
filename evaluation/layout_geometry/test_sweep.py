"""Analytical and small integration checks for the camera-layout sweep."""

import unittest

from .sweep import (
    CAMERA_NAMES, CASES, CLEARANCE, FLOOR, FOCAL, HALF_RIG, LENGTH, NORMS, POINTS, POLE, SAMPLES,
    camera_centers, camera_orientations, cuboid_visible, euler, evaluate, hand_visibility, np, pole_blocks,
    project_and_info, statistics,
)


class LayoutTests(unittest.TestCase):
    def test_distances_and_names(self):
        equal = camera_centers([1, -1, 1], 'equal_radius')
        cube = camera_centers([1, -1, 1], 'cube_vertex')
        np.testing.assert_allclose(equal[:3], [[0, -.65, 0], [.65, 0, 0], [0, 0, .65]])
        np.testing.assert_allclose(equal[:3], cube[:3])
        self.assertAlmostEqual(np.linalg.norm(equal[-1]), HALF_RIG)
        self.assertAlmostEqual(np.linalg.norm(cube[-1]), np.sqrt(3) * HALF_RIG)
        self.assertEqual(CAMERA_NAMES, ('front', 'side', 'top', 'diagonal'))

    def test_axis_aligned_visible_counts(self):
        for signs, expected in [([1, -1, 1], (7, 7, 9, 9)), ([-1, 1, -1], (8, 7, 12, 9))]:
            for mode in ('equal_radius', 'cube_vertex'):
                centers = camera_centers(signs, mode)
                counts = cuboid_visible(centers, np.eye(3), np.zeros(3)).sum(axis=0)
                corners = counts[:8]
                edge_samples = counts[8:].reshape(12, SAMPLES)
                actual = tuple(int(np.sum(values)) for values in (
                    corners >= 1, corners >= 2, (edge_samples >= 1).all(axis=1), (edge_samples >= 2).all(axis=1)))
                self.assertEqual(actual, expected)

    def test_same_edge_fragment_and_information_order(self):
        mask = np.zeros((1, 4, len(POINTS)), dtype=bool)
        mask[0, 0, 8] = True
        mask[0, 1, 9] = True
        bare = statistics(mask, np.zeros((4, len(POINTS), 6, 6)))
        self.assertFalse(np.any(bare['all_support'] >= 2))
        self.assertFalse(bare['full_rank'].any())
        self.assertTrue(np.isinf(bare['condition']).all())

        centers = camera_centers([-1, 1, -1], 'equal_radius')
        orientation = euler([.1, .22, -.38])
        translation = np.array([.02, -.04, .005])
        world = POINTS @ orientation.T + translation
        frame, info = project_and_info(centers, camera_orientations(centers), world, translation)
        geometric = frame & cuboid_visible(centers, orientation, translation)
        visibility = hand_visibility(centers, world, orientation, translation, .008) & geometric
        np.testing.assert_array_equal(visibility[0], geometric)
        self.assertFalse(np.any(visibility[1:] & ~geometric))
        result = statistics(visibility, info)
        self.assertTrue(np.all(result['spectrum'][:, :1] + 1e-8 >= result['spectrum'][:, 1:]))

    def test_image_information_matches_finite_differences(self):
        centers = camera_centers([1, -1, 1], 'cube_vertex')
        camera_r = camera_orientations(centers)
        origin = np.array([.05, .01, -.02])
        points = POINTS @ euler([-.1, .2, .3]).T + origin
        _, info = project_and_info(centers, camera_r, points, origin)
        step = 1e-6

        def image(point, delta):
            changed = origin + LENGTH * delta[:3] + euler(delta[3:]) @ (point - origin)
            optical = camera_r[0] @ (changed - centers[0])
            return FOCAL * optical[:2] / optical[2]

        def derivative(point):
            return np.column_stack([(image(point, delta) - image(point, -delta)) / (2 * step)
                                    for delta in np.eye(6) * step])

        corner_jacobian = derivative(points[0])
        np.testing.assert_allclose(info[0, 0], corner_jacobian.T @ corner_jacobian / NORMS, rtol=1e-6, atol=1e-6)
        # The first edge joins corners 0 and 1. Its sample contributes only the normal residual.
        direction = image(points[1], np.zeros(6)) - image(points[0], np.zeros(6))
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        edge_jacobian = normal @ derivative(points[8])
        np.testing.assert_allclose(info[0, 8], np.outer(edge_jacobian, edge_jacobian) / (NORMS * SAMPLES),
                                   rtol=1e-6, atol=1e-6)

    def test_pole_and_floor_geometry(self):
        start = np.array([[POLE[0] - .2, POLE[1], 0], [POLE[0], POLE[1], .9]])
        finish = np.array([[POLE[0] + .2, POLE[1], 0], [POLE[0], POLE[1], .8], [POLE[0], POLE[1], .25]])
        blocked = pole_blocks(start, finish)
        self.assertTrue(blocked[0, 0])
        self.assertFalse(blocked[1, 1])
        self.assertTrue(blocked[1, 2])
        self.assertLess(camera_centers([1, 1, -1], 'cube_vertex')[-1, 2], FLOOR + CLEARANCE)
        self.assertGreater(camera_centers([1, 1, -1], 'equal_radius')[-1, 2], FLOOR + CLEARANCE)

    def test_small_sweep_reports_every_omission(self):
        rows, drops = evaluate(3)
        self.assertEqual(len(rows), 2 * 8 * len(CASES) * 16)
        self.assertEqual(len(drops), len(rows) * 4)
        floor = next(row for row in rows if (row['octant'], row['distance_mode'], row['scenario'], row['hand']) ==
                     ('x+y+z-', 'cube_vertex', 'floor_clearance', 'none'))
        self.assertAlmostEqual(floor['edge_sample_gain_over_axes_2plus_pp'], 0)
        same_drop = next(row for row in drops if (row['octant'], row['distance_mode'], row['scenario'], row['hand'], row['omitted']) ==
                         ('x+y+z-', 'cube_vertex', 'floor_clearance', 'none', 'diagonal'))
        self.assertAlmostEqual(floor['full_lambda_min_p10'], same_drop['lambda_min_p10'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
