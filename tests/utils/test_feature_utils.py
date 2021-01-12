"""Unit test for common feature utils."""
import unittest

import numpy as np

from gtsam import Cal3Bundler

import gtsfm.utils.features as feature_utils


class TestFeatureUtils(unittest.TestCase):
    """Class containing all unit tests for feature utils."""

    def test_normalize_coordinates(self):
        coordinates = np.array([[10.0, 20.0], [25.0, 12.0], [30.0, 33.0]])

        intrinsics = Cal3Bundler(fx=100, k1=0.0, k2=0.0, u0=20.0, v0=30.0)

        normalized_coordinates = feature_utils.normalize_coordinates(
            coordinates, intrinsics
        )

        expected_coordinates = np.array(
            [[-0.1, -0.1], [0.05, -0.18], [0.1, 0.03]]
        )

        np.testing.assert_allclose(normalized_coordinates, expected_coordinates)
