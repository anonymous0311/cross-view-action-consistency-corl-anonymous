"""Data alignment and held-out camera support invariants."""

import unittest

import numpy as np

from afcv.robot_data import nearest_timestamps
from afcv.support import classify_camera, sample_restricted_camera


class DataTests(unittest.TestCase):
    def test_timestamp_threshold_and_ties(self):
        indices, valid = nearest_timestamps([0.0, 0.1, 0.3, 0.667, 0.8], [0.05, 0.15, 0.6])
        np.testing.assert_array_equal(indices, [0, 1, 1, 2, 2])
        np.testing.assert_array_equal(valid, [True, True, False, True, False])
        # Exactly equidistant binary fractions choose the earlier sample.
        self.assertEqual(nearest_timestamps([0.25], [0.0, 0.5], 1)[0][0], 0)

    def test_reject_unsorted_timestamps(self):
        with self.assertRaises(ValueError):
            nearest_timestamps([0.1, 0.0], [0.0, 0.1])

    def test_restricted_training_excludes_held_out_bands(self):
        rng = np.random.default_rng(42)
        for _ in range(2000):
            self.assertEqual(classify_camera(sample_restricted_camera(rng))[1], "support")


if __name__ == "__main__":
    unittest.main()
