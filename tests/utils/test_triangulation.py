"""Unit tests for triangulation utils.

Authors: Ayush Baid
"""
import unittest

import numpy as np
from gtsam import PinholeCameraCal3Bundler, Pose3, Rot3

import gtsfm.utils.triangulation as triangulation_utils


class TestTriangulationUtils(unittest.TestCase):
    def test_calculate_triangulation_angle_in_degrees(self):
        """Test the computation of triangulation angle using a simple example."""

        camera_center_1 = np.array([0, 0, 0])
        camera_center_2 = np.array([10, 0, 0])
        point_3d = np.array([5, 0, 10])

        expected = 53.1301024
        computed = triangulation_utils.calculate_triangulation_angle_in_degrees(
            camera_1=PinholeCameraCal3Bundler(Pose3(Rot3(), camera_center_1)),
            camera_2=PinholeCameraCal3Bundler(Pose3(Rot3(), camera_center_2)),
            point_3d=point_3d,
        )
        self.assertAlmostEqual(computed, expected)


if __name__ == "__main__":
    unittest.main()
