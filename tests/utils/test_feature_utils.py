"""
Unit test for common feature utils.
"""
import unittest

import numpy as np

import utils.feature_utils as feature_utils


class TestFeatureUtils(unittest.TestCase):
    """
    Class containing all the unit tests.
    """

    def test_convert_to_opencv_keypoints(self):
        """
        Unit tests for numpy feature to opencv conversion.
        """

        numpy_features = np.array([
            [1.3, 5, 1, 4.2],
            [20, 10, 5, 3.2]
        ])

        results = feature_utils.convert_to_opencv_keypoints(numpy_features)

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
