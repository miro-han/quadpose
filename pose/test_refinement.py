"""CPU regression tests for correspondence, normalized information and frame fallback."""

import itertools
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from pose.fusion import (
    EDGE_AXES, BoxModel, Detections, Pose, build_problem, rotation_vector, visible,
)
from pose.synthetic_benchmark import camera_rig
from pose import refinement as ref


class AssignmentTests(unittest.TestCase):
    def test_hungarian_matches_exhaustive_partial_assignment(self):
        rng = np.random.default_rng(31)
        for _ in range(20):
            costs = rng.uniform(0, 12, size=(4, 3))
            costs[rng.random(costs.shape) < 0.25] = np.inf
            output = ref.hungarian_pairs(costs, 9.)
            objective = costs[tuple(output.T)].sum() + 9 * (4 - len(output))
            reference = np.inf
            for columns in itertools.product((-1, 0, 1, 2), repeat=4):
                selected = [c for c in columns if c >= 0]
                if len(selected) != len(set(selected)):
                    continue
                reference = min(reference, sum(9 if c < 0 else costs[r, c] for r, c in enumerate(columns)))
            self.assertAlmostEqual(float(objective), float(reference))

    def test_finite_overlap_rejects_collinear_outside_model(self):
        predicted = np.array([[[0., 0.], [80., 0.]]])
        measured = np.array([[[90., 0.], [130., 0.]], [[20., 2.], [50., 2.]],
                             [[20., 1.], [80., 30.]], [[20., 12.], [50., 12.]]])
        result = ref.edge_gate(predicted, measured, np.array([0]), ref.Config())
        np.testing.assert_array_equal(np.isfinite(result.cost[0]), [False, True, False, False])

    def test_topology_rejects_confident_nonincident_endpoint(self):
        predicted = observed = np.array([[[0., 0.], [80., 0.]]])
        landmarks = np.array([7]), np.array([[0., 0.]])
        config = ref.Config()
        plain = ref.edge_gate(predicted, observed, np.array([0]), replace(config, topology=False), landmarks)
        constrained = ref.edge_gate(predicted, observed, np.array([0]), config, landmarks)
        self.assertTrue(np.isfinite(plain.cost[0, 0]))
        self.assertTrue(np.isinf(constrained.cost[0, 0]))

    def test_fragments_merge_to_one_model_line_and_duplicates_are_unmatched(self):
        predicted = np.array([[[0., 0.], [90., 0.]]])
        observed = np.array([[[0., 0.], [31., 0.]], [[37., 0.], [74., 0.]], [[1., 0.], [29., 0.]]])
        config = ref.Config()
        gate = ref.edge_gate(predicted, observed, np.array([0]), config)
        pairs = ref.add_fragments(np.array([[0, 0]]), gate, config)
        np.testing.assert_array_equal(pairs, [[0, 0], [0, 1]])
        bundle, edges = ref.bundle_fragments(Detections(np.empty((0, 2)), observed), pairs)
        np.testing.assert_array_equal(edges, [[0, 0]])
        np.testing.assert_allclose(np.sort(bundle.segments[0, :, 0]), [0., 74.])


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.box = BoxModel((0.19, 0.12, 0.085))
        self.rig = camera_rig(np.random.default_rng(6), True)
        self.truth = Pose(rotation_vector(np.array([0.19, -0.13, 0.19])), np.array([0.01, 0.02, -0.01]))
        self.prior = Pose(rotation_vector(np.array([0.195, -0.12, 0.195])), np.array([0.012, 0.018, -0.011]))
        self.detected = tuple(Detections(points, edges) for _, points, _, edges in
                              (visible(self.box, self.truth, camera) for camera in self.rig))

    def test_one_and_two_camera_frames_can_be_observable(self):
        for count in (1, 2, 4):
            for method in ("baseline", "removal", "reassignment", "alternating"):
                result = ref.refine_frame(self.prior, self.rig[-count:], self.detected[-count:], self.box,
                                          method=method, assignment="hungarian")
                self.assertTrue(result.valid, (count, method, result.reason))
                self.assertEqual(result.diagnostics.rank, 6)
                np.testing.assert_allclose(ref.pose_delta(result.pose, self.truth), 0, atol=0.01)
                for used, verified in zip(result.edge_pairs, result.verified_edge_pairs, strict=True):
                    self.assertTrue(set(map(tuple, verified)) <= set(map(tuple, used)))
                accepted = [round(row.objective, 3) for row in result.history if row.accepted]
                self.assertTrue(all(b <= a + 0.00101 for a, b in zip(accepted, accepted[1:])))

    def test_parallel_edges_stay_unobservable_with_four_cameras(self):
        raw_detections = []
        raw_pairs = []
        for camera in self.rig:
            _, _, edge_ids, edges = visible(self.box, self.truth, camera)
            select = EDGE_AXES[edge_ids] == 0
            raw_detections.append(Detections(np.empty((0, 2)), edges[select]))
            raw_pairs.append((ref.EMPTY, np.column_stack((edge_ids[select], np.arange(select.sum())))))
        matches = ref.view_problems(self.rig, raw_detections, self.box, raw_pairs)
        diagnostic, _ = ref.final_observability(self.truth, self.box, matches, raw_pairs, ref.Config())
        self.assertLess(diagnostic.rank, 6)
        self.assertTrue(np.isinf(diagnostic.condition))
        self.assertGreater(diagnostic.edge_direction_counts[:, 0].sum(), 4)
        self.assertEqual(diagnostic.edge_direction_counts[:, 1:].sum(), 0)

    def test_normalized_jacobian_agrees_with_object_centred_perturbations(self):
        pairs = [ref.associate(self.truth, camera, detection, self.box)
                 for camera, detection in zip(self.rig, self.detected, strict=True)]
        observations = ref.view_problems(self.rig, self.detected, self.box, pairs)
        problem = build_problem(self.box, observations)
        _, jacobian = problem.evaluate(self.truth, jacobian=True)
        analytically = ref.normalized_jacobian(jacobian, self.truth, self.box)
        step = 1e-6
        box_size = np.linalg.norm(self.box.size_m)
        for axis, direction in enumerate(np.eye(6)):
            mm, rad = box_size * step * direction[:3], step * direction[3:]
            ahead = Pose(rotation_vector(rad) @ self.truth.rotation, self.truth.translation + mm)
            back = Pose(rotation_vector(-rad) @ self.truth.rotation, self.truth.translation - mm)
            finite = (problem.evaluate(ahead) - problem.evaluate(back)) / (2 * step)
            np.testing.assert_allclose(analytically[:, axis], finite, atol=1e-6)

    def test_deadline_does_not_mislabel_hold_as_valid(self):
        config = replace(ref.Config(), budget_ms=1e-9)
        result = ref.refine_frame(self.prior, self.rig, self.detected, self.box, last_valid=self.truth, config=config)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "budget_exhausted")
        self.assertIs(result.pose, self.truth)
        self.assertFalse(result.history[0].accepted)
        self.assertGreater(result.stage_ms["total"], 0)

    def test_reassignment_cycle_preserves_last_accepted_fit(self):
        doubled = tuple(Detections(np.repeat(item.points, 2, axis=0), np.repeat(item.segments, 2, axis=0))
                        for item in self.detected)
        calls = 0

        def forced_pairs(prior, camera, raw, box, **kwargs):
            nonlocal calls
            offset = calls // len(self.rig) % 2
            calls += 1
            points, _, edges, _ = visible(box, self.truth, camera)
            return (np.column_stack((points, 2 * np.arange(len(points)) + offset)),
                    np.column_stack((edges, 2 * np.arange(len(edges)) + offset)))

        with patch.object(ref, "associate", side_effect=forced_pairs):
            result = ref.refine_frame(self.truth, self.rig, doubled, self.box)
        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "cycle")
        self.assertEqual(len(result.history), 2)
        self.assertTrue(result.history[-1].accepted)
        self.assertEqual(result.history[-1].correspondence_changes, 1.)

    def test_over_pruning_preserves_last_accepted_fit(self):
        with patch.object(ref, "verify_pairs", return_value=(ref.EMPTY, ref.EMPTY)):
            result = ref.refine_frame(self.truth, self.rig, self.detected, self.box, method="removal")
        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "over_pruned")
        self.assertGreater(sum(len(item) for item in result.edge_pairs), 0)
        self.assertFalse(result.history[-1].accepted)


if __name__ == "__main__":
    unittest.main()
