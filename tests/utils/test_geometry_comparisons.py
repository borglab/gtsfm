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
        rz = np.deg2rad(90)
        wRi2 = Rot3.RzRyRx(0, 0, rz)  # x-axis of i2 points along y in world frame
        wTi2_estimated = Pose3(wRi2, Point3(0, 0, 0))
        wTi1_estimated = Pose3(Rot3(), Point3(-1, 0, 0))  # At (0, 1, 0) in i2 frame, rotation of i1 is irrelevant here.
        i2Ui1_measured = Unit3(Point3(1, 0, 0))
        # Estimated relative translation of i1 in i2 frame is (0, 1, 0), and the measurement in i2 frame is (1, 0, 0).
        # Expected angle between the two is 90 degrees.
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


def test_get_points_within_radius_of_cameras():
    """Verify that points that fall outside of 10 meter radius of two camera poses.

    Cameras are placed at (0,0,0) and (10,0,0).
    """
    wTi0 = Pose3(Rot3(), np.zeros(3))
    wTi1 = Pose3(Rot3(), np.array([10.0, 0, 0]))
    wTi_list = [wTi0, wTi1]
    points_3d = np.array([[-15, 0, 0], [0, 15, 0], [-5, 0, 0], [15, 0, 0], [25, 0, 0]])
    radius = 10.0
    nearby_points_3d = geometry_comparisons.get_points_within_radius_of_cameras(wTi_list, points_3d, radius)

    expected_nearby_points_3d = np.array(
        [
            [-5, 0, 0],
            [15, 0, 0],
        ]
    )
    np.testing.assert_allclose(nearby_points_3d, expected_nearby_points_3d)


def test_get_points_within_radius_of_cameras_negative_radius():
    """Catch degenerate input."""
    wTi0 = Pose3(Rot3(), np.zeros(3))
    wTi1 = Pose3(Rot3(), np.array([10.0, 0, 0]))
    wTi_list = [wTi0, wTi1]
    points_3d = np.array([[-15, 0, 0], [0, 15, 0], [-5, 0, 0], [15, 0, 0], [25, 0, 0]])
    radius = -5
    nearby_points_3d = geometry_comparisons.get_points_within_radius_of_cameras(wTi_list, points_3d, radius)
    assert nearby_points_3d is None, "Non-positive radius is not allowed"


def test_get_points_within_radius_of_cameras_no_points():
    """Catch degenerate input."""

    wTi0 = Pose3(Rot3(), np.zeros(3))
    wTi1 = Pose3(Rot3(), np.array([10.0, 0, 0]))
    wTi_list = [wTi0, wTi1]
    points_3d = np.zeros((0, 3))
    radius = 10.0

    nearby_points_3d = geometry_comparisons.get_points_within_radius_of_cameras(wTi_list, points_3d, radius)
    assert nearby_points_3d is None, "At least one 3d point must be provided"


def test_get_points_within_radius_of_cameras_no_poses():
    """Catch degenerate input."""
    wTi_list = []
    points_3d = np.array([[-15, 0, 0], [0, 15, 0], [-5, 0, 0], [15, 0, 0], [25, 0, 0]])
    radius = 10.0

    nearby_points_3d = geometry_comparisons.get_points_within_radius_of_cameras(wTi_list, points_3d, radius)
    assert nearby_points_3d is None, "At least one camera pose must be provided"


if __name__ == "__main__":
    unittest.main()
