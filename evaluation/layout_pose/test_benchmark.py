from types import SimpleNamespace
import unittest

from evaluation.layout_pose import benchmark as bench

import numpy as np


class LayoutBenchmarkTests(unittest.TestCase):
    def test_camera_center_distances_and_fixed_axis_calibration(self):
        all_centers = []
        for signs in bench.DIAGONALS:
            equal = bench.camera_rig("equal_radius", signs)
            cube = bench.camera_rig("cube_vertex", signs)
            np.testing.assert_allclose(equal[-1].center * np.sqrt(3), cube[-1].center)
            self.assertAlmostEqual(np.linalg.norm(equal[-1].center), 0.65)
            self.assertAlmostEqual(np.linalg.norm(cube[-1].center), 0.65 * np.sqrt(3))
            for first, second in zip(equal[:3], cube[:3], strict=True):
                np.testing.assert_array_equal(first.center, second.center)
                np.testing.assert_array_equal(first.rotation, second.rotation)
            for camera in (*equal, *cube):
                pixel, depth = bench.project(np.zeros((1, 3)), bench.Pose(np.eye(3), np.zeros(3)), camera)
                np.testing.assert_allclose(pixel[0], [480, 360], atol=1e-12)
                self.assertGreater(depth[0], 0)
            all_centers.append(tuple(cube[-1].center))
        self.assertEqual(len(set(all_centers)), 8)

    def test_sphere_ray_tests_finite_segment_not_infinite_line(self):
        camera = np.array([0., 0., 0.])
        endpoints = np.array([[3., 0., 0.], [0., 3., 0.], [0.4, 0., 0.]])
        blocked = bench.sphere_blocks(camera, endpoints, np.array([1., 0., 0.]), 0.2)
        np.testing.assert_array_equal(blocked, [True, False, False])
        np.testing.assert_array_equal(bench.sphere_blocks(camera, endpoints, np.array([-1., 0., 0.]), 0.2), False)

    def test_noise_pairing_keeps_physical_edge_draws_consistent(self):
        first = bench.noise_for(13, 3, 0)
        second = bench.noise_for(13, 3, 0)
        changed_camera = bench.noise_for(13, 3, 1)
        np.testing.assert_array_equal(first.edge_pixels, second.edge_pixels)
        self.assertFalse(np.array_equal(first.edge_pixels, changed_camera.edge_pixels))

    def test_pair_agreement_requires_two_correct_physical_ids(self):
        detections = bench.Detections(np.empty((0, 2)), np.empty((1, 2, 2)))
        samples = (bench.EMPTY, *(bench.Sample(detections, np.empty(0, int), np.array([item])) for item in (0, 2, 0)))
        pair = SimpleNamespace(segment_pairs=np.array([[0, 0]]))
        matches = (SimpleNamespace(segment_pairs=np.empty((0, 2), int)), pair, pair, pair)
        data = tuple(SimpleNamespace(edge_ids=np.array([0]), usable_edges=np.array([view != 0])) for view in range(4))
        result = bench.evaluate_edges(samples, matches, data, 0, "hand")
        self.assertEqual(result["multiview_pairs"], 3)
        self.assertEqual(result["multiview_pairs_correct"], 1)
        self.assertEqual(result["designated_edges_lost"], 1)
        self.assertEqual(result["designated_edges_correct_elsewhere"], 1)

    def test_missing_diagonal_pose_is_exactly_paired_across_modes(self):
        first = bench.evaluate(13, 2, "equal_radius", (-1, -1, -1))
        second = bench.evaluate(13, 2, "cube_vertex", (1, -1, 1))
        for a, b in zip(first, second, strict=True):
            if a["condition"] == "missing_diagonal":
                for key in ("translation_mm", "rotation_deg", "valid", "failure", "edges_correct"):
                    self.assertEqual(a[key], b[key])

    def test_three_seed_sequences_cover_fourteen_hand_directions(self):
        for seed in bench.SEEDS:
            poses, prior = bench.trajectory(seed, 28)
            directions = {tuple(bench.hand_for(pose, seed, frame)[1]) for frame, pose in enumerate(poses)}
            self.assertEqual(len(directions), 14)
            self.assertLess(bench.errors(prior, poses[0])[0], 20)


if __name__ == "__main__":
    unittest.main()
