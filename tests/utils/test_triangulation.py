"""Unit tests for triangulation utils.

Authors: Ayush Baid, John Lambert
"""
import unittest

import numpy as np
from gtsam import PinholeCameraCal3Bundler, Pose3, Rot3

import gtsfm.utils.triangulation as triangulation_utils


class TestTriangulationUtils(unittest.TestCase):
    def test_calculate_triangulation_angle_in_degrees(self) -> None:
        """Test the computation of triangulation angle using a simple example.

        Cameras and point situated as follows in the x-z plane:

        (0,0,0)
        ____ +z
        | 
        |
        +x          
                   X (5,0,5)
                        
        (10,0,0)
        ____ +z
        |
        |
        +x
        """
        camera_center_1 = np.array([0, 0, 0])
        camera_center_2 = np.array([10, 0, 0])
        point_3d = np.array([5, 0, 5])

        expected = 90

        wT0 = Pose3(Rot3(), camera_center_1)
        wT1 = Pose3(Rot3(), camera_center_2)
        computed = triangulation_utils.calculate_triangulation_angle_in_degrees(
            camera_1=PinholeCameraCal3Bundler(wT0),
            camera_2=PinholeCameraCal3Bundler(wT1),
            point_3d=point_3d,
        )
        self.assertAlmostEqual(computed, expected)


if __name__ == "__main__":
    unittest.main()