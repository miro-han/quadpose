"""Policy states must be causal and transitions must execute only selected actions."""
import unittest
from unittest.mock import patch

import numpy as np

from learning.datasets import iter_trials
from learning.policy import State, action_mask, features, transition


class PolicyTests(unittest.TestCase):
    def test_predict_skips_solver_and_preserves_measurements(self):
        trial = next(iter_trials("train", regimes=("clean",), repeats=1))
        data = trial.frames[0].detections
        state = State(((-1/30, trial.initial),))
        with patch("learning.policy.evaluate_hypotheses", side_effect=AssertionError("solver called")):
            after, _, kind, _ = transition(state, "predict", 0., trial.rig, data, trial.box)
        self.assertEqual(kind, "predicted")
        self.assertEqual(after.misses, 1)
        self.assertIs(after.history, state.history)
        self.assertFalse(np.array_equal(features(state, data), features(after, data)))

    def test_all_does_not_generate_triples(self):
        trial = next(iter_trials("train", regimes=("clean",), repeats=1))
        data = trial.frames[0].detections
        state = State(((-1/30, trial.initial),))
        self.assertTrue(action_mask(data)[0])
        with patch("learning.policy.evaluate_hypotheses", side_effect=AssertionError("generated triples")):
            after, _, status, _ = transition(state, "all", 0., trial.rig, data, trial.box)
        self.assertEqual(status, "measured")
        self.assertIsNot(after.history, state.history)


if __name__ == "__main__":
    unittest.main()

