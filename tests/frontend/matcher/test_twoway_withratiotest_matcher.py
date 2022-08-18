"""Tests for the two-way-with-ratio-test matcher.

Authors: Ayush Baid
"""
import unittest

import numpy as np

import gtsfm.utils.features as feature_utils
import tests.frontend.matcher.test_matcher_base as test_matcher_base
from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher


class TestTwoWayWithRatioTestMatcher(test_matcher_base.TestMatcherBase):
    """Unit tests for the TwoWayMatcher configured to use the ratio-test.

    All unit test functions defined in TestMatcherBase are run automatically.
    """

    def setUp(self):
        super().setUp()

        self.matcher = TwoWayMatcher(ratio_test_threshold=0.8)

    def test_on_dummy_data(self):
        """Test using dummy 1D descriptors to verify correctness."""

        image_shape_i1 = (300, 100)  # as (H,W)
        descriptors_i1 = (
            np.array([0.4865, 0.3752, 0.3077, 0.9188, 0.7837, 0.1083, 0.6822, 0.3764, 0.2288, 0.8018, 1.1])
            .reshape(-1, 1)
            .astype(np.float32)
        )
        keypoints_i1 = feature_utils.generate_random_keypoints(descriptors_i1.shape[0], image_shape_i1)

        image_shape_i2 = (300, 100)  # as (H,W)
        descriptors_i2 = (
            np.array([0.9995, 0.3376, 0.9005, 0.5382, 0.3162, 0.7974, 0.1785, 0.3491, 0.8658, 0.2912])
            .reshape(-1, 1)
            .astype(np.float32)
        )
        keypoints_i2 = feature_utils.generate_random_keypoints(descriptors_i2.shape[0], image_shape_i2)
        expected_matches = np.array([[9, 5], [2, 4], [3, 2], [0, 3]])

        result = self.matcher.match(
            keypoints_i1, keypoints_i2, descriptors_i1, descriptors_i2, image_shape_i1, image_shape_i2
        )
        print(result)
        np.testing.assert_array_equal(result, expected_matches)


if __name__ == "__main__":
    unittest.main()
