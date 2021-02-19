"""Unit tests for comparison functions for geometry types.

Authors: Ayush Baid
"""
import unittest

import numpy as np
from gtsam import Cal3_S2, Point3, Pose3, Rot3, Unit3
from gtsam.examples import SFMdata
from scipy.spatial.transform import Rotation

import gtsfm.utils.geometry_comparisons as geometry_comparisons

POSE_LIST = SFMdata.createPoses(Cal3_S2())


class TestGeometryComparisons(unittest.TestCase):
    """Unit tests for comparison functions for geometry types."""

    def test_compare_poses_exact(self):
        """Check pose comparison with exactly same inputs."""
        self.assertTrue(geometry_comparisons.compare_global_poses(POSE_LIST, POSE_LIST))

    def test_compare_poses_with_uniform_scaled_translations(self):
        """Check pose comparison with all translations in input #2 scaled by
        the same scalar factor."""
        scale_factor = 1.2
        pose_list_ = [Pose3(x.rotation(), x.translation() * scale_factor) for x in POSE_LIST]

        self.assertTrue(geometry_comparisons.compare_global_poses(POSE_LIST, pose_list_))

    def test_compare_poses_with_nonuniform_scaled_translations(self):
        """Check pose comparison with all translations in input #2 scaled by
        significantly different scalar factors."""
        scale_factors = [0.3, 0.7, 0.9, 1.0, 1.0, 0.99, 1.01, 1.10]
        pose_list_ = [Pose3(x.rotation(), x.translation() * scale_factors[idx]) for idx, x in enumerate(POSE_LIST)]

        self.assertFalse(geometry_comparisons.compare_global_poses(POSE_LIST, pose_list_))

    def test_compare_poses_with_origin_shift(self):
        """Check pose comparison with a shift in the global origin."""
        new_origin = Pose3(Rot3.RzRyRx(0.3, 0.1, -0.27), np.array([-20.0, +19.0, 3.5]))

        pose_list_ = [new_origin.between(x) for x in POSE_LIST]

        self.assertTrue(geometry_comparisons.compare_global_poses(POSE_LIST, pose_list_))

    def test_compare_different_poses(self):
        """Compare pose comparison with different inputs."""

        pose_list = [POSE_LIST[1], POSE_LIST[2], POSE_LIST[3]]
        pose_list_ = [POSE_LIST[2], POSE_LIST[3], POSE_LIST[1]]

        self.assertFalse(geometry_comparisons.compare_global_poses(pose_list, pose_list_))

    def test_compute_relative_rotation_angle(self):
        """Tests the relative angle between two rotations."""

        R_1 = Rot3.RzRyRx(0, np.deg2rad(45), np.deg2rad(22.5))
        R_2 = Rot3.RzRyRx(0, np.deg2rad(90), np.deg2rad(22.5))

        # returns angle in degrees
        computed_deg = geometry_comparisons.compute_relative_rotation_angle(R_1, R_2)
        expected_deg = 45

        np.testing.assert_allclose(computed_deg, expected_deg, rtol=1e-3, atol=1e-3)

    def test_compute_relative_unit_translation_angle(self):
        """Tests the relative angle between two unit-translations."""

        U_1 = Unit3(np.array([1, 0, 0]))
        U_2 = Unit3(np.array([0.5, 0.5, 0]))

        # returns angle in degrees
        computed_deg = geometry_comparisons.compute_relative_unit_translation_angle(U_1, U_2)
        expected_deg = 45

        self.assertAlmostEqual(computed_deg, expected_deg, places=3)

    def test_compare_global_poses_scaled_squares(self):
        """Make sure a big and small square can be aligned.

        The u's represent a big square (10x10), and v's represents a small square (4x4).
        """
        R0 = Rotation.from_euler("z", 0, degrees=True).as_matrix()
        R90 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        R180 = Rotation.from_euler("z", 180, degrees=True).as_matrix()
        R270 = Rotation.from_euler("z", 270, degrees=True).as_matrix()

        wTu0 = Pose3(Rot3(R0), np.array([2, 3, 0]))
        wTu1 = Pose3(Rot3(R90), np.array([12, 3, 0]))
        wTu2 = Pose3(Rot3(R180), np.array([12, 13, 0]))
        wTu3 = Pose3(Rot3(R270), np.array([2, 13, 0]))

        wTi_list = [wTu0, wTu1, wTu2, wTu3]

        wTv0 = Pose3(Rot3(R0), np.array([4, 3, 0]))
        wTv1 = Pose3(Rot3(R90), np.array([8, 3, 0]))
        wTv2 = Pose3(Rot3(R180), np.array([8, 7, 0]))
        wTv3 = Pose3(Rot3(R270), np.array([4, 7, 0]))

        wTi_list_ = [wTv0, wTv1, wTv2, wTv3]

        pose_graphs_equal = geometry_comparisons.compare_global_poses(wTi_list, wTi_list_)
        self.assertTrue(pose_graphs_equal)

    def test_compute_translation_to_direction_angle_is_zero(self):
        i2Ui1_measured = Unit3(Point3(1, 0, 0))
        wTi2_estimated = Pose3(Rot3(), Point3(0, 0, 0))
        wTi1_estimated = Pose3(Rot3(), Point3(2, 0, 0))
        self.assertEqual(
            geometry_comparisons.compute_translation_to_direction_angle(i2Ui1_measured, wTi2_estimated, wTi1_estimated),
            0.0,
        )

    def test_compute_translation_to_direction_angle_is_nonzero(self):
        i2Ui1_measured = Unit3(Point3(0, 1, 0))
        wRi2 = Rot3.RzRyRx(-np.deg2rad(90), 0, 0)  # x-axis points to -y in world frame
        wRi1 = Rot3.RzRyRx(np.deg2rad(30), np.deg2rad(60), np.deg2rad(90))  # irrelevant
        wTi2_estimated = Pose3(wRi2, Point3(0, 0, 0))
        wTi1_estimated = Pose3(Rot3(), Point3(0, -1, 0))  # along -y axis
        # estimated direction along x-axis and measured along y-axis in i2 frame.
        self.assertTrue(
            geometry_comparisons.compute_translation_to_direction_angle(i2Ui1_measured, wTi2_estimated, wTi1_estimated),
            90.0,
        )

    def test_compute_points_distance_l2_is_zero(self):
        self.assertEqual(
            geometry_comparisons.compute_points_distance_l2(Point3(1, -2, 3), Point3(1, -2, 3)),
            0.0,
        )

    def test_compute_points_distance_l2_is_none(self):
        self.assertEqual(
            geometry_comparisons.compute_points_distance_l2(Point3(0, 0, 0), None),
            None,
        )

    def test_compute_points_distance_l2_is_nonzero(self):
        wti1 = Point3(1, 1, 1)
        wti2 = Point3(1, 1, -1)
        self.assertEqual(geometry_comparisons.compute_points_distance_l2(wti1, wti2), 2)


if __name__ == "__main__":
    unittest.main()
