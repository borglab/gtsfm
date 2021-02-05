"""Unit tests for comparison functions for geometry types.

Authors: Ayush Baid
"""
import unittest

import numpy as np
from gtsam import Cal3_S2, Point3, Pose3, Rot3
from gtsam.examples import SFMdata
from scipy.spatial.transform import Rotation

import gtsfm.utils.geometry_comparisons as geometry_comparisons

POSE_LIST = SFMdata.createPoses(Cal3_S2())


class TestGeometryComparisons(unittest.TestCase):
    """Unit tests for comparison functions for geometry types."""

    def test_compare_poses_exact(self):
        """Check pose comparison with exactly same inputs."""
        self.assertTrue(
            geometry_comparisons.compare_global_poses(POSE_LIST, POSE_LIST)
        )

    def test_compare_poses_with_uniform_scaled_translations(self):
        """Check pose comparison with all translations in input #2 scaled by
        the same scalar factor."""
        scale_factor = 1.2
        pose_list_ = [
            Pose3(x.rotation(), x.translation() * scale_factor)
            for x in POSE_LIST
        ]

        self.assertTrue(
            geometry_comparisons.compare_global_poses(POSE_LIST, pose_list_)
        )

    def test_compare_poses_with_nonuniform_scaled_translations(self):
        """Check pose comparison with all translations in input #2 scaled by
        significantly different scalar factors."""
        scale_factors = [0.3, 0.7, 0.9, 1.0, 1.0, 0.99, 1.01, 1.10]
        pose_list_ = [
            Pose3(x.rotation(), x.translation() * scale_factors[idx])
            for idx, x in enumerate(POSE_LIST)
        ]

        self.assertFalse(
            geometry_comparisons.compare_global_poses(POSE_LIST, pose_list_)
        )

    def test_compare_poses_with_origin_shift(self):
        """Check pose comparison with a shift in the global origin."""
        new_origin = Pose3(
            Rot3.RzRyRx(0.3, 0.1, -0.27), np.array([-20.0, +19.0, 3.5])
        )

        pose_list_ = [new_origin.between(x) for x in POSE_LIST]

        self.assertTrue(
            geometry_comparisons.compare_global_poses(POSE_LIST, pose_list_)
        )

    def test_compare_different_poses(self):
        """Compare pose comparison with different inputs."""

        pose_list = [POSE_LIST[1], POSE_LIST[2], POSE_LIST[3]]
        pose_list_ = [POSE_LIST[2], POSE_LIST[3], POSE_LIST[1]]

        self.assertFalse(
            geometry_comparisons.compare_global_poses(pose_list, pose_list_)
        )

    def test_compare_global_poses_scaled_squares(self):
        """Make sure a big and small square can be aligned.

        The u's represent a big square (10x10), and v's represents a small square (4x4).
        """
        R0 = Rotation.from_euler("z", 0, degrees=True).as_matrix()
        R90 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        R180 = Rotation.from_euler("z", 180, degrees=True).as_matrix()
        R270 = Rotation.from_euler("z", 270, degrees=True).as_matrix()

        wTu0 = Pose3(Rot3(R0), Point3(np.array([2, 3, 0])))
        wTu1 = Pose3(Rot3(R90), Point3(np.array([12, 3, 0])))
        wTu2 = Pose3(Rot3(R180), Point3(np.array([12, 13, 0])))
        wTu3 = Pose3(Rot3(R270), Point3(np.array([2, 13, 0])))

        wTi_list = [wTu0, wTu1, wTu2, wTu3]

        wTv0 = Pose3(Rot3(R0), Point3(np.array([4, 3, 0])))
        wTv1 = Pose3(Rot3(R90), Point3(np.array([8, 3, 0])))
        wTv2 = Pose3(Rot3(R180), Point3(np.array([8, 7, 0])))
        wTv3 = Pose3(Rot3(R270), Point3(np.array([4, 7, 0])))

        wTi_list_ = [wTv0, wTv1, wTv2, wTv3]

        pose_graphs_equal = geometry_comparisons.compare_global_poses(
            wTi_list, wTi_list_
        )
        self.assertTrue(pose_graphs_equal)


if __name__ == "__main__":
    unittest.main()
