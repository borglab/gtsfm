"""Unit tests for comparison functions for geometry types.

Authors: Ayush Baid
"""

import unittest
from unittest.mock import patch

import numpy as np
from gtsam import Point3, Pose3, Rot3, Unit3
from gtsam.examples import SFMdata

import gtsfm.utils.geometry_comparisons as geometry_comparisons

POSE_LIST = SFMdata.posesOnCircle(R=40)

ROT3_EULER_ANGLE_ERROR_THRESHOLD = 1e-2
POINT3_RELATIVE_ERROR_THRESH = 1e-1
POINT3_ABS_ERROR_THRESH = 1e-2


def rot3_compare(R: Rot3, R_: Rot3, msg=None) -> bool:
    return np.allclose(R.xyz(), R_.xyz(), atol=1e-2)


def point3_compare(t: Point3, t_: Point3, msg=None) -> bool:
    return np.allclose(t, t_, rtol=POINT3_RELATIVE_ERROR_THRESH, atol=POINT3_ABS_ERROR_THRESH)


class TestGeometryComparisons(unittest.TestCase):
    """Unit tests for comparison functions for geometry types."""

    def setUp(self):
        super().setUp()

        self.addTypeEqualityFunc(Rot3, rot3_compare)
        self.addTypeEqualityFunc(Point3, point3_compare)

    @patch(
        "gtsfm.utils.align.align_rotations",
        return_value=[
            Rot3.RzRyRx(0, np.deg2rad(32), 0),
            Rot3.RzRyRx(0, 0, np.deg2rad(-22)),
            Rot3.RzRyRx(0, 0, np.deg2rad(83)),
        ],  # compared with aRi_list
    )
    def test_compare_rotations_with_all_valid_rot3s_success(self, align_rotations_mocked):
        """Tests the comparison results on list of rotations."""

        aRi_list = [
            Rot3.RzRyRx(0, np.deg2rad(25), 0),
            Rot3.RzRyRx(0, 0, np.deg2rad(-20)),
            Rot3.RzRyRx(0, 0, np.deg2rad(80)),
        ]
        bRi_list = [
            Rot3.RzRyRx(0, np.deg2rad(31), 0),
            Rot3.RzRyRx(0, 0, np.deg2rad(-22)),
            Rot3.RzRyRx(0, 0, np.deg2rad(77.5)),
        ]  # meaningless as align function is mocked

        # test with threshold of 10 degrees, which satisfies all the rotations.
        self.assertTrue(geometry_comparisons.compare_rotations(aRi_list, bRi_list, 10))
        align_rotations_mocked.assert_called_once()

    @patch(
        "gtsfm.utils.align.align_rotations",
        return_value=[
            Rot3.RzRyRx(0, np.deg2rad(32), 0),
            Rot3.RzRyRx(0, 0, np.deg2rad(-22)),
            Rot3.RzRyRx(0, 0, np.deg2rad(83)),
        ],  # compared with aRi_list
    )
    def test_compare_rotations_with_all_valid_rot3s_failure(self, align_rotations_mocked):
        """Tests the comparison results on list of rotations."""

        aRi_list = [
            Rot3.RzRyRx(0, np.deg2rad(25), 0),
            Rot3.RzRyRx(0, 0, np.deg2rad(-20)),
            Rot3.RzRyRx(0, 0, np.deg2rad(80)),
        ]
        bRi_list = [
            Rot3.RzRyRx(0, np.deg2rad(31), 0),
            Rot3.RzRyRx(0, 0, np.deg2rad(-22)),
            Rot3.RzRyRx(0, 0, np.deg2rad(77.5)),
        ]  # meaningless as align function is mocked

        # test with threshold of 5 degrees, which fails one rotation and hence the overall comparison
        self.assertFalse(geometry_comparisons.compare_rotations(aRi_list, bRi_list, 5))
        align_rotations_mocked.assert_called_once()

    @patch(
        "gtsfm.utils.align.align_rotations",
        return_value=[Rot3.RzRyRx(0, np.deg2rad(25), 0), Rot3.RzRyRx(0, 0, np.deg2rad(-20))],  # compared with aRi_list
    )
    def test_compare_rotations_with_nones_at_same_indices(self, align_rotations_mocked):
        """Tests the comparison results on list of rotations."""

        list1 = [
            Rot3.RzRyRx(0, np.deg2rad(25), 0),
            Rot3.RzRyRx(0, 0, np.deg2rad(-20)),
            None,
        ]
        list2 = [
            Rot3.RzRyRx(0, np.deg2rad(31), 0),
            Rot3.RzRyRx(0, 0, np.deg2rad(-22)),
            None,
        ]
        threshold_degrees = 10

        # test with threshold of 10 degrees, which satisfies all the rotations.
        self.assertTrue(geometry_comparisons.compare_rotations(list1, list2, threshold_degrees))
        align_rotations_mocked.assert_called_once()

    @patch("gtsfm.utils.align.align_rotations", return_value=None)
    def test_compare_rotations_with_nones_at_different_indices(self, aligned_rotations_mocked):
        """Tests the comparison results on list of rotations."""

        list1 = [
            Rot3.RzRyRx(0, np.deg2rad(25), 0),
            Rot3.RzRyRx(0, 0, np.deg2rad(-20)),
            None,
        ]
        list2 = [
            Rot3.RzRyRx(0, np.deg2rad(31), 0),
            None,
            Rot3.RzRyRx(0, 0, np.deg2rad(-22)),
        ]

        # test with threshold of 10 degrees, which satisfies all the rotations.
        self.assertFalse(geometry_comparisons.compare_rotations(list1, list2, 10))
        aligned_rotations_mocked.assert_not_called()

    def test_compute_relative_rotation_angle(self):
        """Tests the relative angle between two rotations."""

        R_1 = Rot3.RzRyRx(0, np.deg2rad(45), np.deg2rad(22.5))
        R_2 = Rot3.RzRyRx(0, np.deg2rad(90), np.deg2rad(22.5))

        # returns angle in degrees
        computed_deg = geometry_comparisons.compute_relative_rotation_angle(R_1, R_2)
        expected_deg = 45

        np.testing.assert_allclose(computed_deg, expected_deg, rtol=1e-3, atol=1e-3)

    def test_compute_relative_rotation_angle2(self) -> None:
        """Tests the relative angle between two rotations

        Currently compute_relative_rotation_angle() uses Scipy, so this test compares with GTSAM.

        TODO(johnwlambert): replace this test with Scipy function calls once we fix the GTSAM's .axisAngle() code.
        """
        num_trials = 1000
        np.random.seed(0)

        def random_rotation() -> Rot3:
            """Sample a random rotation by generating a sample from the 4d unit sphere."""
            q = np.random.randn(4)
            # make unit-length quaternion
            q /= np.linalg.norm(q)
            qw, qx, qy, qz = q
            R = Rot3(qw, qx, qy, qz)
            return R

        for _ in range(num_trials):
            # generate 2 random rotations
            wR1 = random_rotation()
            wR2 = random_rotation()

            computed_deg = geometry_comparisons.compute_relative_rotation_angle(wR1, wR2)

            i2Ri1 = wR2.between(wR1)
            _, expected_rad = i2Ri1.axisAngle()
            expected_deg = np.rad2deg(expected_rad)

        np.testing.assert_allclose(computed_deg, expected_deg, rtol=1e-3, atol=1e-3)

    def test_compute_relative_unit_translation_angle(self):
        """Tests the relative angle between two unit-translations."""

        U_1 = Unit3(np.array([1, 0, 0]))
        U_2 = Unit3(np.array([0.5, 0.5, 0]))

        # returns angle in degrees
        computed_deg = geometry_comparisons.compute_relative_unit_translation_angle(U_1, U_2)
        expected_deg = 45

        self.assertAlmostEqual(computed_deg, expected_deg, places=3)

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
            geometry_comparisons.compute_points_distance_l2(wti1=Point3(1, -2, 3), wti2=Point3(1, -2, 3)), 0.0
        )

    def test_compute_points_distance_l2_is_none(self):
        self.assertEqual(geometry_comparisons.compute_points_distance_l2(wti1=Point3(0, 0, 0), wti2=None), None)

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

    expected_nearby_points_3d = np.array([[-5, 0, 0], [15, 0, 0]])
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


def test_compute_cyclic_rotation_error() -> None:
    """Ensure cycle error is computed correctly within a triplet.

    Imagine 3 poses, all centered at the origin, at different orientations.

    Ground truth poses:
       Let i0 face along +x axis (0 degrees in yaw)
       Let i2 have a 30 degree rotation from the +x axis.
       Let i4 have a 90 degree rotation from the +x axis.

    However, suppose one edge measurement is corrupted (from i0 -> i4) by 5 degrees.
    """
    i2Ri0 = Rot3.Ry(np.deg2rad(30))
    i4Ri2 = Rot3.Ry(np.deg2rad(60))
    i4Ri0 = Rot3.Ry(np.deg2rad(95))

    cycle_error = geometry_comparisons.compute_cyclic_rotation_error(i2Ri0, i4Ri2, i4Ri0)
    assert np.isclose(cycle_error, 5)


def test_is_valid_SO3() -> None:
    """Ensures that rotation matrices are accurately checked for SO(3) membership."""
    R = Rot3()
    assert geometry_comparisons.is_valid_SO3(R)

    # fmt: off
    # Determinant and diagonal are obviously not unit-sized.
    R = np.array(
        [
            [3.85615, 0.0483263, 1.5018],
            [-1.50233, 0.199159, 3.8511],
            [-0.0273012, -4.13347, 0.203112]
        ]
    )
    R = Rot3(R)
    # fmt: on
    assert not geometry_comparisons.is_valid_SO3(R)


if __name__ == "__main__":
    unittest.main()
