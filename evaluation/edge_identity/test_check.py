"""Checks for finite edge verification and physical-ID accounting."""

from types import SimpleNamespace
import unittest

from .check import additional_pair_counts, gate_geometry, np


class EdgeIdentityTests(unittest.TestCase):
    def test_trimmed_and_reversed_observations_pass(self):
        predicted = np.tile([[[0., 0.], [100., 0.]]], (2, 1, 1))
        measured = np.array([[[20., 1.], [80., 1.]], [[80., 1.], [20., 1.]]])
        np.testing.assert_array_equal(gate_geometry(predicted, measured), [True, True])

    def test_distance_and_finite_overlap_reject_independently(self):
        predicted = np.tile([[[0., 0.], [100., 0.]]], (3, 1, 1))
        measured = np.array([[[20., 3.], [80., 3.]], [[75., 0.], [125., 0.]], [[78., 0.], [98., 0.]]])
        np.testing.assert_array_equal(gate_geometry(predicted, measured), [False, False, True])

    def test_angle_rejects_short_crossing_segment_near_line(self):
        predicted = np.array([[[0., 0.], [10., 0.]]])
        measured = np.array([[[1., -0.8], [9., .8]]])
        self.assertFalse(gate_geometry(predicted, measured)[0])

    def test_false_pair_clutter_and_wrong_physical_are_disjoint(self):
        samples = tuple(SimpleNamespace(edge_labels=np.array(labels)) for labels in ([0], [0], [2], [-1]))
        claims = tuple(np.array([[0, 0]]) for _ in samples)
        self.assertEqual(additional_pair_counts(samples, claims),
                         dict(detected_pair_opportunities=1, false_pairs_with_clutter=3, false_pairs_physical_only=2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
