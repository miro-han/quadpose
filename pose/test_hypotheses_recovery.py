"""Regression tests for the Phase IV cross-view and causal recovery boundaries."""

from dataclasses import replace
import unittest

import cv2
import numpy as np

from pose.fusion import analyze_frame, choose_result

from evaluation.pose_benchmark.bench_recovery import BOX, CAMERAS, EMPTY, motion, observe
from .hypotheses import evaluate_hypotheses, pose_distance, select_pose
from .recovery import RecoveryConfig, RecoveryTracker, predict_pose


class CrossViewTests(unittest.TestCase):
    def test_legacy_and_independent_holdout(self):
        prior, truth = motion(0), motion(1)
        clean, bad = observe(truth, seed=11), observe(truth, fault=0, seed=11)
        baseline = analyze_frame(prior, CAMERAS, bad, BOX)
        before = evaluate_hypotheses(prior, CAMERAS, clean, BOX)
        after = evaluate_hypotheses(prior, CAMERAS, bad, BOX)
        self.assertEqual(len(after.triples), 4)
        np.testing.assert_allclose(before.triples[0].fit.pose.translation, after.triples[0].fit.pose.translation)
        np.testing.assert_allclose(before.triples[0].fit.pose.rotation, after.triples[0].fit.pose.rotation)
        self.assertTrue(np.isfinite(after.triples[0].held_out_residual_px))
        self.assertGreater(after.triples[0].held_out_residual_px, 10.)
        self.assertFalse(after.triples[0].supported[0])
        for policy in ('all', 'best'):
            outcome = select_pose(after, policy)
            legacy = choose_result(baseline, policy)
            self.assertEqual(outcome.valid, legacy.valid)
            np.testing.assert_allclose(outcome.pose.translation, legacy.pose.translation)
            np.testing.assert_allclose(outcome.pose.rotation, legacy.pose.rotation)

    def test_failed_holdout_cannot_veto_and_claims_preserve_observation_ids(self):
        prior, truth = motion(0), motion(1)
        raw = observe(truth, fault=0, seed=21)
        matrix = evaluate_hypotheses(prior, CAMERAS, raw, BOX)
        for policy in ('cross', 'cross_verify', 'cross_alternating'):
            with self.subTest(policy=policy):
                result = select_pose(matrix, policy)
                self.assertTrue(result.valid)
                self.assertFalse(result.support[0])
                self.assertEqual(result.diagnostics['omitted'], 0)
                self.assertLess(pose_distance(result.pose, truth)[0], .003)
                for pairs, frame in zip(result.segment_pairs, raw, strict=True):
                    self.assertTrue(np.all(pairs[:, 1] < len(frame.segments)))
                    self.assertTrue(np.all(pairs[:, 1] >= 0))


class RecoveryTests(unittest.TestCase):
    def test_prediction_stops_and_cannot_emit_claims(self):
        tracker = RecoveryTracker(BOX, motion(0), mode='velocity',
                                  config=RecoveryConfig(prediction_seconds=.15))
        for index in (0, 1):
            status = tracker.step(.05 * index, CAMERAS, observe(motion(index)))
            self.assertEqual(status.status, 'measured')
        expected = predict_pose(tracker.history, .10)
        predicted = tracker.step(.10, CAMERAS, (EMPTY,) * 4)
        self.assertEqual(predicted.status, 'predicted')
        self.assertFalse(predicted.valid)
        np.testing.assert_allclose(predicted.pose.translation, expected.translation)
        self.assertTrue(all(len(pairs) == 0 for pairs in predicted.segment_pairs))
        self.assertEqual(len(tracker.history), 2)
        lost = tracker.step(.25, CAMERAS, (EMPTY,) * 4)
        self.assertEqual(lost.status, 'lost')
        self.assertIsNone(lost.pose)
        with self.assertRaises(ValueError):
            tracker.step(.25, CAMERAS, (EMPTY,) * 4)

    def test_partial_and_parallel_observability(self):
        truth = motion(1)
        for count in (1, 2):
            with self.subTest(count=count):
                tracker = RecoveryTracker(BOX, truth, mode='partial')
                result = tracker.step(0., CAMERAS, observe(truth, keep=range(count)))
                self.assertEqual(result.status, 'measured')
                self.assertEqual(result.diagnostics['rank'], 6)
                self.assertEqual(int(result.support.sum()), count)
        parallel = tuple(replace(CAMERAS[3], name=f'p{i}') for i in range(4))
        observations = observe(truth, cameras=parallel, parallel=True)
        matrix = evaluate_hypotheses(truth, parallel, observations, BOX)
        self.assertLess(matrix.all.rank, 6)
        self.assertFalse(select_pose(matrix, 'cross').valid)

    def test_image_only_pnp_reacquires_without_pose_seed(self):
        truth = motion(0, jump=True)
        result = RecoveryTracker(BOX, None, mode='pnp').step(0., CAMERAS, observe(truth, seed=90))
        self.assertEqual(result.reason, 'image_pnp_reinitialized')
        self.assertTrue(result.reacquired)
        self.assertGreater(result.diagnostics['pnp_proposals'], 0)
        distance, angle = pose_distance(result.pose, truth)
        self.assertLess(distance, .01)
        self.assertLess(angle, 3.)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    unittest.main()
