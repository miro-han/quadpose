import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from pose.fusion import AssociationLabels, BoxModel, Detections, Pose, build_problem, solve_pose, view_match, visible
from pose.refinement import EdgeGate

from .datasets import make_rig
from .track_a import fragment_frame
from .track_a_geometry import Candidates, learned_pairs, select_rank, solve_weighted
from .track_a_models import fit


class LearningTests(unittest.TestCase):
    def test_binary_and_shared_utility_follow_synthetic_order(self):
        points = np.array([[-3., 0.], [-1., 0.], [1., 0.], [3., 0.]])
        binary = fit(points, np.array([0, 0, 1, 1]), steps=150)
        np.testing.assert_array_equal(binary.probability(points) > .5, [False, False, True, True])
        np.testing.assert_array_equal(binary.center, points.mean(axis=0))
        ranking = fit(points, np.array([0, 0, 1, 1]), paired=points[::-1], hidden_width=4, steps=160)
        self.assertLess(ranking.forward(points)[0], ranking.forward(points)[-1])

    def test_unmatched_and_fragments_are_not_forced_into_one_to_one_result(self):
        cost = np.zeros((1, 3))
        intervals = np.array([[[0., 12.], [18., 35.], [10., 30.]]])
        candidates = Candidates(np.array([4]), EdgeGate(cost, intervals), np.array([[.95], [.83], [.19]]),
                                np.array([0, 0, 0]), np.array([0, 1, 2]), np.empty((0, 2), int))
        scorer = SimpleNamespace(probability=lambda features: features[:, 0])
        _, edges = learned_pairs(candidates, scorer, .5)
        np.testing.assert_array_equal(edges, [[4, 0], [4, 1]])

    def test_fragment_stress_transforms_clutter_like_true_edges(self):
        view = Detections(np.empty((0, 2)), np.array([[[0., 0.], [50., 0.]], [[0., 1.], [50., 1.]]]))
        labels = AssociationLabels(np.empty(0, int), np.array([3, -1]))
        frame = MockFrame((view,), (labels,))
        altered = fragment_frame(frame)
        np.testing.assert_array_equal(altered.labels[0].edge_ids, [3, 3, -1, -1])
        self.assertLess(altered.detections[0].segments[0, 1, 0], altered.detections[0].segments[1, 0, 0])

    def test_weighted_huber_with_unit_confidence_matches_established_fit(self):
        box, rig = BoxModel((.19, .12, .085)), make_rig()
        truth = Pose(np.eye(3), np.array([.015, -.012, .02]))
        prior = Pose(np.eye(3), truth.translation + [.005, -.002, .004])
        matches = []
        for camera in rig:
            corners, points, edges, lines = visible(box, truth, camera)
            matches.append(view_match(camera, Detections(points, lines), np.column_stack((corners, np.arange(len(corners)))),
                                      np.column_stack((edges, np.arange(len(edges))))))
        problem = build_problem(box, matches)
        expected = solve_pose(prior, problem)
        actual = solve_weighted(prior, problem, np.ones(len(problem.vertices)))
        self.assertTrue(expected.valid and actual.valid)
        np.testing.assert_allclose(expected.pose.translation, actual.pose.translation, atol=1e-9)
        np.testing.assert_allclose(expected.pose.rotation, actual.pose.rotation, atol=1e-9)

    def test_learned_ranker_cannot_select_masked_candidate(self):
        matrix = SimpleNamespace(triples=(None, None, None, None))
        scorer = SimpleNamespace(forward=lambda features: np.array([20., 1., 2., 0.]))
        self.assertEqual(select_rank(matrix, np.zeros((4, 2)), [False, True, True, False], "rank_mlp", scorer), 2)
        self.assertIsNone(select_rank(matrix, np.zeros((4, 2)), [False] * 4, "rank_mlp", scorer))


@dataclass
class MockFrame:
    detections: tuple
    labels: tuple


if __name__ == "__main__":
    unittest.main()
