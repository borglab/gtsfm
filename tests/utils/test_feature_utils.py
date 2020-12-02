"""
Unit test for common feature utils.
"""
import unittest

import numpy as np

import utils.features as feature_utils


class TestFeatureUtils(unittest.TestCase):
    """
    Class containing all the unit tests.
    """

    def test_keypoints_from_array(self):
        """
        Unit tests for conversion of keypoints from numpy array representation
        to opencv's keypoints.
        """

        numpy_features = np.array([
            [1.3, 5, 1, 4.2],
            [20, 10, 5, 3.2]
        ])

        results = feature_utils.keypoints_from_array(numpy_features)

        # Check the length of the result
        self.assertEqual(numpy_features.shape[0], len(results))

        # check the first keypoint
        kp = results[0]
        self.assertAlmostEqual(numpy_features[0][0], kp.pt[0])
        self.assertAlmostEqual(numpy_features[0][1], kp.pt[1])
        self.assertAlmostEqual(numpy_features[0][2], kp.size)

        # check the second keypoint
        kp = results[1]
        self.assertAlmostEqual(numpy_features[1][0], kp.pt[0])
        self.assertAlmostEqual(numpy_features[1][1], kp.pt[1])
        self.assertAlmostEqual(numpy_features[1][2], kp.size)
