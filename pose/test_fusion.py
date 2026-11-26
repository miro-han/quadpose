import unittest
from dataclasses import replace

import numpy as np

from pose.fusion import (
    TOKEN_FIELDS, BoxModel, Detections, Pose, advance, analyze_frame, associate_view,
    build_problem, choose_result, make_tokens, view_match, visible,
)
from pose.synthetic_benchmark import (
    REGIMES, camera_rig, evaluate_trial, generate_trial, longest_uncovered_segment, pose_error, trial_specs,
)


class PhaseTwoTests(unittest.TestCase):
    def setUp(self):
        self.box = BoxModel((0.19, 0.12, 0.085))
        self.cameras = camera_rig(np.random.default_rng(19), True)
        self.truth = Pose(np.eye(3), np.array([0.02, -0.01, 0.025]))
        self.prior = advance(self.truth, np.array([0.012, -0.009, 0.005, 0.001, 0.002, -0.002]))
        self.detections = tuple(Detections(points, edges) for _, points, _, edges in
                                (visible(self.box, self.truth, camera) for camera in self.cameras))
        self.empty = Detections(np.empty((0, 2)), np.empty((0, 2, 2)))

    def test_noiseless_joint_fit_and_token_contract(self):
        evidence = analyze_frame(self.prior, self.cameras, self.detections, self.box)
        np.testing.assert_array_equal(evidence.support, np.ones((4, 4), dtype=bool))
        self.assertEqual(evidence.residuals.shape, (4, 4))
        self.assertTrue(np.allclose(evidence.residuals, 0, atol=1e-4))
        tokens = make_tokens(evidence)
        self.assertEqual(tokens.shape, (4, len(TOKEN_FIELDS)))
        self.assertTrue(np.isfinite(tokens).all())
        for method in ("all", "best", "average", "reliable"):
            estimate = choose_result(evidence, method)
            self.assertTrue(estimate.valid)
            np.testing.assert_allclose(pose_error(estimate.pose, self.truth), 0, atol=1e-3)
            self.assertEqual(estimate.support.sum(), 3 if method == "best" else 4)

    def test_one_missing_view_keeps_only_the_real_triple(self):
        detections = (*self.detections[:3], self.empty)
        evidence = analyze_frame(self.prior, self.cameras, detections, self.box)
        self.assertEqual([fit.valid for fit in evidence.hypotheses], [False, False, False, True])
        self.assertTrue(np.isinf(evidence.residuals[3, 3]))
        np.testing.assert_array_equal(evidence.support[3], [True, True, True, False])
        for method in ("best", "average", "reliable"):
            estimate = choose_result(evidence, method)
            self.assertTrue(estimate.valid)
            self.assertEqual(estimate.excluded, (3,))

    def test_two_missing_views_invalidate_triple_fusion(self):
        detections = (*self.detections[:2], self.empty, self.empty)
        evidence = analyze_frame(self.prior, self.cameras, detections, self.box)
        self.assertFalse(any(fit.valid for fit in evidence.hypotheses))
        for method in ("best", "average", "reliable"):
            estimate = choose_result(evidence, method)
            self.assertFalse(estimate.valid)
            self.assertIs(estimate.pose, self.prior)

    def test_places_single_coherently_shifted_view_outside_support(self):
        displaced = Detections(self.detections[0].points + [14, 3], self.detections[0].segments + [14, 3])
        evidence = analyze_frame(self.prior, self.cameras, (displaced, *self.detections[1:]), self.box)
        estimate = choose_result(evidence)
        self.assertTrue(estimate.valid)
        self.assertEqual(estimate.excluded, (0,))
        self.assertGreater(evidence.residuals[0, 0], 5.5)
        np.testing.assert_allclose(pose_error(estimate.pose, self.truth), 0, atol=1e-3)

    def test_pixel_jacobian_agrees_with_pose_updates(self):
        camera, detections = self.cameras[0], self.detections[0]
        matches = view_match(camera, detections, *associate_view(self.prior, camera, detections, self.box))
        problem = build_problem(self.box, [matches])
        _, jacobian = problem.evaluate(self.prior, jacobian=True)
        for axis, delta in enumerate(1e-6 * np.eye(6)):
            finite = (problem.evaluate(advance(self.prior, delta)) - problem.evaluate(advance(self.prior, -delta))) / 2e-6
            np.testing.assert_allclose(jacobian[:, axis], finite, atol=1e-6)

    def test_external_occluder_removes_only_covered_portion(self):
        segment = np.array([[-4., 0.], [8., 0.]])
        uncovered = longest_uncovered_segment(segment, [-1., -1.], [2., 1.])
        np.testing.assert_allclose(uncovered, [[2., 0.], [8., 0.]])
        fully_covered = longest_uncovered_segment(segment, [-5., -1.], [9., 1.])
        self.assertEqual(np.linalg.norm(fully_covered[1] - fully_covered[0]), 0)

    def test_split_trajectories_and_rng_are_independent_and_reproducible(self):
        splits = ("train", "validation", "test")
        schedules = [trial_specs(split, 1, 16) for split in splits]
        self.assertEqual({spec.regime for spec in schedules[0]}, set(REGIMES))
        self.assertEqual(len({spec.seed for specs in schedules for spec in specs}), len(splits) * len(REGIMES))
        samples = [generate_trial(specs[-1]) for specs in schedules]
        self.assertEqual(len({trial.motion_family for trial in samples}), len(splits))
        for trial in samples:
            self.assertTrue(any(frame.truth.fault_mask.sum() == 2 for frame in trial.frames))
            self.assertEqual(np.count_nonzero(trial.frames[0].truth.fault_mask), 0)
        replica = generate_trial(schedules[-1][-1])
        np.testing.assert_array_equal(replica.frames[12].detections[0].points, samples[-1].frames[12].detections[0].points)

    def test_changing_truth_does_not_change_estimated_packet(self):
        trial = generate_trial(trial_specs("test", 1, 16)[0])
        frame = trial.frames[0]
        changed_truth = replace(frame.truth, pose=advance(frame.truth.pose, np.array([0, 0, 0, 0.1, 0, 0])))
        rows, tokens = evaluate_trial(replace(trial, frames=(frame,)))
        changed_rows, changed_tokens = evaluate_trial(replace(trial, frames=(replace(frame, truth=changed_truth),)))
        for key, data in tokens[0].items():
            if key == "tokens" or key.startswith("packet_"):
                np.testing.assert_array_equal(data, changed_tokens[0][key])
        self.assertNotEqual(rows[0]["translation_mm"], changed_rows[0]["translation_mm"])


if __name__ == "__main__":
    unittest.main()
