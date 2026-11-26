import unittest

import numpy as np

from experiments.fusion.fusion_poc import (
    EDGES, SCENARIOS, VERTICES, Detections, Pose, cameras, estimated_associations, fit_pose,
    increment, initial_pose, observations, pose_error, project, run_trial, simulate, trajectory,
    visible_projections,
)


class FusionTests(unittest.TestCase):
    def test_rig_has_paired_diagonal_and_fixed_axis_cameras(self):
        same, opposite = cameras("same"), cameras("opposite")
        for first, second in zip(same[:3], opposite[:3], strict=True):
            np.testing.assert_array_equal(first.center, second.center)
            np.testing.assert_array_equal(first.rotation, second.rotation)
        np.testing.assert_array_equal(same[3].center, -opposite[3].center)
        np.testing.assert_array_equal(same[3].center, np.sum([camera.center for camera in same[:3]], axis=0))
        for camera in same[:3]:
            self.assertAlmostEqual(np.linalg.norm(camera.center), abs(same[3].center[0]))
        for camera in (*same, *opposite):
            centered = camera.rotation @ -camera.center
            np.testing.assert_allclose(centered[:2], 0, atol=1e-12)
            self.assertGreater(centered[2], 0)
            self.assertAlmostEqual(np.linalg.det(camera.rotation), 1)

    def test_projection_jacobian(self):
        camera = cameras()[3]
        pose = trajectory(12)
        ids = np.arange(len(VERTICES))
        _, jacobian = project(pose, camera, ids, jacobian=True)
        for axis in range(6):
            delta = np.eye(6)[axis] * 1e-6
            plus = project(increment(pose, delta), camera, ids)[0]
            minus = project(increment(pose, -delta), camera, ids)[0]
            np.testing.assert_allclose(jacobian[:, :, axis], (plus - minus) / 2e-6, atol=1e-6)

    def test_centered_box_gains_coverage_from_opposite_diagonal(self):
        centered = Pose(np.eye(3), np.zeros(3))
        for layout, expected in (("same", (7, 7, 9, 9)), ("opposite", (8, 7, 12, 9))):
            visible = [visible_projections(centered, camera) for camera in cameras(layout)]
            counts = []
            for feature_field in (0, 2):
                views = np.bincount(np.concatenate([camera[feature_field] for camera in visible]))
                counts.extend([np.count_nonzero(views >= 1), np.count_nonzero(views >= 2)])
            self.assertEqual(tuple(counts), expected)

    def test_feature_only_pose_fits_converge_on_noiseless_data(self):
        true_pose = trajectory(0)
        for method in ("point", "edge", "hybrid"):
            inputs = []
            for camera in cameras():
                point_ids, points, edge_ids, segments = visible_projections(true_pose, camera)
                trimmed = 0.75 * segments + 0.25 * segments[:, ::-1]
                point_pairs = np.column_stack((point_ids, np.arange(len(points))))
                edge_pairs = np.column_stack((edge_ids, np.arange(len(segments))))
                inputs.extend(observations(
                    camera, Detections(points, trimmed),
                    point_pairs if method != "edge" else point_pairs[:0],
                    edge_pairs if method != "point" else edge_pairs[:0],
                ))
            estimate, solved = fit_pose(initial_pose(), inputs)
            self.assertTrue(solved)
            np.testing.assert_allclose(pose_error(estimate, true_pose), 0, atol=1e-4)

    def test_estimated_pairs_are_one_to_one_and_endpoint_order_invariant(self):
        camera = cameras()[-1]
        pose = trajectory(5)
        detections, _ = simulate(pose, camera, SCENARIOS["baseline"], "baseline", 5, np.random.default_rng(0))
        pairs = estimated_associations(pose, camera, detections)
        reversed_pairs = estimated_associations(pose, camera, Detections(detections.points, detections.segments[:, ::-1]))
        for matches, flipped in zip(pairs, reversed_pairs, strict=True):
            np.testing.assert_array_equal(matches, flipped)
            for ids in matches.T:
                self.assertEqual(len(set(ids)), len(ids))

    def test_rank_failure_holds_prior(self):
        prior = initial_pose()
        estimate, solved = fit_pose(prior, [])
        self.assertFalse(solved)
        self.assertIs(estimate, prior)

    def test_frame_metrics_do_not_count_oracle_mismatches(self):
        rows, coverage = run_trial(2, 42, "baseline", "opposite")
        self.assertEqual(len(rows), 24)
        self.assertEqual(len(coverage), 2 * (len(VERTICES) + len(EDGES)))
        self.assertTrue(all(row["wrong_matches"] == 0 for row in rows if row["association"] == "oracle"))
        self.assertTrue(all(row["detected_views"] <= row["geometric_views"] <= 4 for row in coverage))

    def test_axis_only_hybrid_is_shared_between_layouts(self):
        same, _ = run_trial(2, 42, "stress", "same", axis_only=True)
        opposite, _ = run_trial(2, 42, "stress", "opposite", axis_only=True)
        self.assertEqual(len(same), 2)
        self.assertEqual(len(opposite), 2)
        for first, second in zip(same, opposite, strict=True):
            self.assertEqual(first, dict(second, layout="same"))


if __name__ == "__main__":
    unittest.main()
