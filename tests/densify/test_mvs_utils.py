"""Unit tests for utilities for MVS methods

Authors: Ren Liu, Ayush Baid
"""
import unittest

import numpy as np
from gtsam import PinholeCameraCal3Bundler, Pose3, Rot3

import gtsfm.densify.mvs_utils as mvs_utils


class TestMVSUtils(unittest.TestCase):
    """Unit tests for utilities for MVS methods."""

    def test_calculate_triangulation_angle_in_degrees(self):
        """Test the computation of triangulation angle using a simple example."""

        camera_center_1 = np.array([0, 0, 0])
        camera_center_2 = np.array([10, 0, 0])
        point_3d = np.array([5, 0, 5])

        expected = 90

        computed = mvs_utils.calculate_triangulation_angle_in_degrees(
            camera_1=PinholeCameraCal3Bundler(Pose3(Rot3(), camera_center_1)),
            camera_2=PinholeCameraCal3Bundler(Pose3(Rot3(), camera_center_2)),
            point_3d=point_3d,
        )
        self.assertAlmostEqual(computed, expected)

    def test_piecewise_gaussian_below_expect_baseline_angle(self) -> None:
        """Unit test for the case that the angle between two coordinates is below the expect baseline angle,
        where sigma_1 is used to calculate the score"""

        score = mvs_utils.piecewise_gaussian(theta=4, theta_0=5, sigma_1=1, sigma_2=10)

        self.assertAlmostEqual(score, np.exp(-(1.0 ** 2) / (2 * 1.0 ** 2)))

    def test_piecewise_gaussian_above_expect_baseline_angle(self) -> None:
        """Unit test for the case that the angle between two coordinates is above the expect baseline angle,
        where sigma_2 is used to calculate the score"""

        score = mvs_utils.piecewise_gaussian(theta=10, theta_0=5, sigma_1=1, sigma_2=10)

        self.assertAlmostEqual(score, np.exp(-(5.0 ** 2) / (2 * 10.0 ** 2)))


if __name__ == "__main__":
    unittest.main()
