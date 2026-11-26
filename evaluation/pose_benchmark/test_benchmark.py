"""Tests of deterministic data, label scoring, and method-local history."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .benchmark import RunResult, feature_counts, recovery_onset, run_trial
from .benchmark_data import config, np, source, trial


class IndependentBenchmarkTests(unittest.TestCase):
    def test_frozen_seeds_are_disjoint_and_repeatable(self):
        settings = config()
        self.assertFalse(set(settings['development_seeds']) & set(settings['test_seeds']))
        options = settings['test_seeds'][0], 'fragmentation', settings['layouts']['upper_rear'], settings
        a, b = trial(*options)[2], trial(*options)[2]
        leaves = [sample.edge_labels for frame in a if frame.active for sample in frame.samples]
        self.assertTrue(any(len(values) != len(set(values)) for values in leaves))
        for left, right in zip(a, b, strict=True):
            for first, second in zip(left.samples, right.samples, strict=True):
                np.testing.assert_array_equal(first.detections.segments, second.detections.segments)
                np.testing.assert_array_equal(first.edge_labels, second.edge_labels)

    def test_blackout_and_parallel_without_corner_evidence(self):
        settings = config()
        seed, layout = settings['test_seeds'][0], settings['layouts']['upper_rear']
        blackout = trial(seed, 'reacquisition', layout, settings)[2]
        during = [row for row in blackout if row.active]
        self.assertEqual(len(during), 7)
        self.assertTrue(all(not len(item.detections.points) and not len(item.detections.segments)
                            for row in during for item in row.samples))
        parallel = trial(seed, 'parallel_edges', layout, settings)[2]
        self.assertTrue(all(not len(item.detections.points) for row in parallel if row.active for item in row.samples))

    def test_pair_denominator_and_fragment_purity(self):
        samples = tuple(SimpleNamespace(edge_labels=np.array(labels, dtype=int)) for labels in ([0, -1], [0], [1], []))
        claims = (np.array([[0, 0], [0, 1]]), np.array([[0, 0]]), np.array([[0, 0]]), np.empty((0, 2), int))
        lost = np.zeros((4, 12), dtype=bool)
        lost[0, 0] = True
        row, mapping = feature_counts(samples, claims, lost, {(0, 0): {0}, (1, 0): {1}})
        expected = dict(edge_assigned=3, edge_correct=1, pair_claims=3, pair_correct=0, pair_available=1,
                        rescue_lost=1, rescue_correct=1, stable_claims=1, stable_eligible=2)
        for key, value in expected.items():
            self.assertEqual(row[key], value)
        self.assertEqual(mapping[1, 0], {0})

    def test_rejected_result_never_updates_method_prior(self):
        settings = config()
        rig, seed, frames = trial(settings['development_seeds'][0], 'clean', settings['layouts']['upper_rear'], settings)
        motion = source.Pose(seed.rotation, np.array([.12, .06, .01]))
        seen = []

        def step(name, previous, _cameras, _samples, _tracker, _stamp):
            seen.append(previous)
            accepted = len(seen) == 2
            return RunResult(motion, accepted, tuple(np.empty((0, 2), int) for _ in rig),
                             'measured' if accepted else 'predicted', '', {})

        with patch('evaluation.pose_benchmark.benchmark.run_method', side_effect=step):
            rows = run_trial(rig, seed, frames[:3], settings, 'example')
        self.assertIs(seen[0], seed)
        self.assertIs(seen[1], seed)
        self.assertIs(seen[2], motion)
        self.assertFalse(rows[0]['success'])

    def test_recovery_requires_three_measured_successes_after_fault(self):
        settings = dict(fault_end_exclusive=4, recovery_consecutive_measured=3)
        rows = [{'success': status} for status in [True, True, True, True, True, False, True, True, True]]
        self.assertEqual(recovery_onset(rows, settings), 2)
        self.assertIsNone(recovery_onset(rows[:-1], settings))


if __name__ == '__main__':
    unittest.main(verbosity=2)
