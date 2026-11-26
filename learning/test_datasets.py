"""Tests for D0 determinism, Layout A, and the inference/target file boundary."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from learning.datasets import (CENTERS, INPUT_FIELDS, OBSERVATION_FIELDS, export_split, iter_trials,
                              load_split)


class DatasetTests(unittest.TestCase):
    def test_disjoint_seed_and_time_families_with_fixed_layout(self):
        groups = {key: list(iter_trials(key, regimes=('clean',), frames=8, repeats=1))
                  for key in ('train', 'dev', 'lockbox')}
        trials = [group[0] for group in groups.values()]
        self.assertEqual(len({trial.seed for trial in trials}), 3)
        self.assertEqual(len({trial.motion for trial in trials}), 3)
        for trial in trials:
            np.testing.assert_array_equal([camera.center for camera in trial.rig], CENTERS)
            self.assertEqual(sum(frame.fault_mask.sum() for frame in trial.frames), 0)
        repeated = next(iter_trials('train', regimes=('clean',), frames=8, repeats=1))
        for raw, repeat in zip(trials[0].frames, repeated.frames, strict=True):
            np.testing.assert_array_equal(raw.truth.rotation, repeat.truth.rotation)
            for left, right in zip(raw.detections, repeat.detections, strict=True):
                np.testing.assert_array_equal(left.segments, right.segments)

    def test_load_need_not_open_target_file(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_split(root / 'train', 'train', frames=8, repeats=1, regimes=('missing',))
            paired = load_split(root, 'train', with_targets=True)
            self.assertEqual(set(paired.inputs), set(INPUT_FIELDS))
            self.assertEqual(set(paired.observations), set(OBSERVATION_FIELDS))
            self.assertTrue(paired.targets['target_fault_mask'].any())
            self.assertEqual(paired.inputs['camera_tokens'].shape, (8, 4, 21))
            self.assertEqual(paired.inputs['hypothesis_tokens'].shape, (8, 4, 15))
            np.testing.assert_array_equal(paired.targets['target_corner_ids'] == -2, ~paired.observations['point_mask'])
            np.testing.assert_array_equal(paired.targets['target_edge_ids'] == -2, ~paired.observations['segment_mask'])
            (root / 'train' / 'targets.npz').rename(root / 'withheld.npz')
            unpaired = load_split(root, 'train')
            self.assertIsNone(unpaired.targets)
            np.testing.assert_array_equal(unpaired.inputs['camera_tokens'], paired.inputs['camera_tokens'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
