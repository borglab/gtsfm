"""Unit tests for comparison functions for geometry types.

Authors: Ayush Baid
"""
import unittest
from typing import List
from unittest.mock import patch

import numpy as np
from gtsam import Cal3_S2, Point3, Pose3, Rot3, Unit3
from gtsam.examples import SFMdata

import gtsfm.utils.geometry_comparisons as geometry_comparisons
import tests.data.sample_poses as sample_poses

POSE_LIST = SFMdata.createPoses(Cal3_S2())

ROT3_EQUALITY_THRESH = 1e-3
POINT3_RELATIVE_ERROR_THRESH = 1e-1
POINT3_ABS_ERROR_THRESH = 1e-2


def rot3_compare(R: Rot3, R_: Rot3, msg=None) -> bool:
    return R.equals(R_, ROT3_EQUALITY_THRESH)


def point3_compare(t: Point3, t_: Point3, msg=None) -> bool:
    return np.allclose(t, t_, rtol=POINT3_RELATIVE_ERROR_THRESH, atol=POINT3_ABS_ERROR_THRESH)


class TestGeometryComparisons(unittest.TestCase):
    """Unit tests for comparison functions for geometry types."""

    def __assert_equality_on_rot3s(self, computed: List[Rot3], expected: List[Rot3]) -> None:

        self.assertEqual(len(computed), len(expected))

        for R, R_ in zip(computed, expected):
            self.assertEqual(R, R_)

    def __assert_equality_on_point3s(self, computed: List[Point3], expected: List[Point3]) -> None:

        self.assertEqual(len(computed), len(expected))

        for t, t_ in zip(computed, expected):
            np.testing.assert_allclose(t, t_, rtol=POINT3_RELATIVE_ERROR_THRESH, atol=POINT3_ABS_ERROR_THRESH)

    def setUp(self):
        super().setUp()

        self.addTypeEqualityFunc(Rot3, rot3_compare)
        self.addTypeEqualityFunc(Point3, point3_compare)

    def test_align_rotations(self):
        """Tests the alignment of rotations."""

        # using rotation along just the Y-axis so that angles can be linearly added.
        input_list = [
            Rot3.RzRyRx(np.deg2rad(0), np.deg2rad(-10), 0),
            Rot3.RzRyRx(np.deg2rad(0), np.deg2rad(30), 0),
        ]
        ref_list = [
            Rot3.RzRyRx(np.deg2rad(0), np.deg2rad(80), 0),
            Rot3.RzRyRx(np.deg2rad(0), np.deg2rad(-40), 0),
        ]

        computed = geometry_comparisons.align_rotations(input_list, ref_list)
        expected = [
            Rot3.RzRyRx(0, np.deg2rad(80), 0),
            Rot3.RzRyRx(0, np.deg2rad(120), 0),
        ]

        self.__assert_equality_on_rot3s(computed, expected)

    def test_align_translations(self):
        """Test for alignment of translations which are located in a circle."""

        translation_shift = np.array([5, 10, -5])
        rotation_shift = Rot3.RzRyRx(0, 0, np.deg2rad(30))
        scaling_factor = 0.7

        input_list = [x.translation() for x in sample_poses.CIRCLE_TWO_EDGES_GLOBAL_POSES]
        ref_list = [scaling_factor * rotation_shift.rotate(x) + translation_shift for x in input_list]

        (
            computed_poses,
            computed_scaling,
            computed_rotation,
            computed_translation,
        ) = geometry_comparisons.align_translations(input_list, ref_list)

        self.assertEqual(
            computed_scaling,
            scaling_factor,
        )
        self.assertTrue(computed_rotation.equals(rotation_shift, ROT3_EQUALITY_THRESH))
        np.testing.assert_allclose(
            computed_translation, translation_shift, rtol=POINT3_RELATIVE_ERROR_THRESH, atol=POINT3_ABS_ERROR_THRESH
        )

        self.__assert_equality_on_point3s(computed_poses, ref_list)

    def test_align_translations_scaled_squares(self):
        """Make sure a big and small square can be aligned.

        The u's represent a big square (10x10), and v's represents a small square (4x4).
        """
        wtu0 = np.array([2, 3, 0])
        wtu1 = np.array([12, 3, 0])
        wtu2 = np.array([12, 13, 0])
        wtu3 = np.array([2, 13, 0])

        input_list = [wtu0, wtu1, wtu2, wtu3]

        wtv0 = np.array([4, 3, 0])
        wtv1 = np.array([8, 3, 0])
        wtv2 = np.array([8, 7, 0])
        wtv3 = np.array([4, 7, 0])

        ref_list = [wtv0, wtv1, wtv2, wtv3]

        aligned_poses, _, _, _ = geometry_comparisons.align_translations(input_list, ref_list)
        self.__assert_equality_on_point3s(aligned_poses, ref_list)

    def test_align_poses(self):
        """Tests for alignment of poses, which are located in a circle."""

        translation_shift = np.array([5, 10, -5])
        rotation_shift_for_translation_model = Rot3.RzRyRx(0, 0, np.deg2rad(30))
        rotation_shift_for_rotation = Rot3.RzRyRx(0, 0, np.deg2rad(-15))
        scaling_factor = 0.7

        input_list = sample_poses.CIRCLE_TWO_EDGES_GLOBAL_POSES
        ref_list = []
        for wTi in input_list:
            wRi = wTi.rotation()
            wti = wTi.translation()

            wRi_ = rotation_shift_for_rotation.compose(wRi)
            wti_ = scaling_factor * rotation_shift_for_translation_model.rotate(wti) + translation_shift
            ref_list.append(Pose3(wRi_, wti_))

        computed = geometry_comparisons.align_poses(input_list, ref_list)
        # TODO: this should ideally be impossible to align w/ correct SIM3 alignment
        self.__assert_equality_on_rot3s([x.rotation() for x in computed], [x.rotation() for x in ref_list])
        self.__assert_equality_on_point3s([x.translation() for x in computed], [x.translation() for x in ref_list])

    def test_compare_rotations_with_all_valid_rot3s(self):
        """Tests the comparison results on list of rotations."""

        list1 = [
            Rot3.RzRyRx(0, np.deg2rad(25), 0),
            Rot3.RzRyRx(0, 0, np.deg2rad(-20)),
            Rot3.RzRyRx(0, 0, np.deg2rad(80)),
        ]
        list2 = [
            Rot3.RzRyRx(0, np.deg2rad(31), 0),
            Rot3.RzRyRx(0, 0, np.deg2rad(-22)),
            Rot3.RzRyRx(0, 0, np.deg2rad(77.5)),
        ]

        # test with threshold of 10 degrees, which satisfies all the rotations.
        self.assertTrue(geometry_comparisons.compare_rotations(list1, list2, 10))

        # test with threshold of 5 degrees, which fails one rotation and hence the overall comparison
        self.assertFalse(geometry_comparisons.compare_rotations(list1, list2, 5))

    def test_compare_rotations_with_nones_at_same_indices(self):
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

        # test with threshold of 10 degrees, which satisfies all the rotations.
        self.assertTrue(geometry_comparisons.compare_rotations(list1, list2, 10))

    def test_compare_rotations_with_nones_at_different_indices(self):
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

    @patch("gtsfm.utils.geometry_comparisons.compare_rotations", return_value=True)
    @patch("gtsfm.utils.geometry_comparisons.align_rotations", return_value=[])
    def test_align_and_compare_rotations(self, align_rotations_mocked, compare_rotations_mocked):

        # mock the align and compare functions and test with Nones at same position
        list1 = [
            Rot3.RzRyRx(0, np.deg2rad(25), 0),
            Rot3.RzRyRx(0, np.deg2rad(-20), 0),
            None,
        ]
        list2 = [
            Rot3.RzRyRx(0, np.deg2rad(40), 0),
            Rot3.RzRyRx(0, np.deg2rad(-20), 0),
            None,
        ]
        angle_threshold = 10
        geometry_comparisons.align_and_compare_rotations(list1, list2, angle_threshold)

        # check calls to the mocked functions
        align_rotations_mocked.assert_called_once()
        compare_rotations_mocked.assert_called_once()

        align_rotations_call_args = align_rotations_mocked.call_args.args

        self.__assert_equality_on_rot3s(
            align_rotations_call_args[0],
            [
                Rot3.RzRyRx(0, np.deg2rad(25), 0),
                Rot3.RzRyRx(0, np.deg2rad(-20), 0),
            ],
        )

        self.__assert_equality_on_rot3s(
            align_rotations_call_args[1],
            [
                Rot3.RzRyRx(0, np.deg2rad(40), 0),
                Rot3.RzRyRx(0, np.deg2rad(-20), 0),
            ],
        )

        compare_rotations_call_args = compare_rotations_mocked.call_args.args
        self.assertEqual(len(compare_rotations_call_args[0]), 0)
        self.__assert_equality_on_rot3s(
            compare_rotations_call_args[1],
            [
                Rot3.RzRyRx(0, np.deg2rad(40), 0),
                Rot3.RzRyRx(0, np.deg2rad(-20), 0),
            ],
        )
        self.assertEqual(compare_rotations_call_args[2], angle_threshold)

    def test_compare_translations_with_all_valid_point3s(self):
        """Tests the comparison results on list of translations."""

        list1 = [
            np.array([0, 20, 0]),
            np.array([0, 0, 12]),
            np.array([0, -5, -6]),
        ]
        list2 = [
            np.array([0, 20.001, 0]),
            np.array([0, 0, 12.005]),
            np.array([0.01, -5, -6]),
        ]

        # test with threshold of 10 degrees, which satisfies all the rotations.
        self.assertTrue(geometry_comparisons.compare_translations(list1, list2, 1e-1, 5e-2))

        # test with threshold of 5 degrees, which fails one rotation and hence the overall comparison
        self.assertFalse(geometry_comparisons.compare_translations(list1, list2, 1e-3, 1e-3))

    def test_compare_translations_with_nones_at_same_indices(self):
        """Tests the comparison results on list of rotations."""

        list1 = [
            np.array([0, 20, 0]),
            np.array([0, 0, 12]),
            None,
        ]
        list2 = [
            np.array([0, 20, 0]),
            np.array([0, 0, 12]),
            None,
        ]

        # test with threshold of 10 degrees, which satisfies all the rotations.
        self.assertTrue(geometry_comparisons.compare_translations(list1, list2, 1e-1, 5e-2))

    def test_compare_translations_with_nones_at_different_indices(self):
        """Tests the comparison results on list of rotations."""

        list1 = [
            np.array([0, 20, 0]),
            np.array([0, 0, 12]),
            None,
        ]
        list2 = [
            np.array([0, 20, 0]),
            None,
            np.array([0, 0, 12]),
        ]

        # test with threshold of 10 degrees, which satisfies all the rotations.
        self.assertFalse(geometry_comparisons.compare_translations(list1, list2, 1e-1, 5e-2))

    @patch("gtsfm.utils.geometry_comparisons.compare_translations", return_value=True)
    @patch("gtsfm.utils.geometry_comparisons.align_translations", return_value=([], None, None, None))
    def test_align_and_compare_translations(self, align_translations_mocked, compare_translations_mocked):

        # mock the align and compare functions and test with Nones at same position
        list1 = [
            np.array([0, 20, 0]),
            np.array([0, 0, 12]),
            None,
        ]
        list2 = [
            np.array([0, 20.0001, 0]),
            np.array([0, 0, 12]),
            None,
        ]
        geometry_comparisons.align_and_compare_translations(
            list1, list2, POINT3_RELATIVE_ERROR_THRESH, POINT3_ABS_ERROR_THRESH
        )

        # check calls to the mocked functions
        align_translations_mocked.assert_called_once()
        compare_translations_mocked.assert_called_once()

        align_translations_call_args = align_translations_mocked.call_args.args

        self.__assert_equality_on_point3s(
            align_translations_call_args[0],
            [
                np.array([0, 20, 0]),
                np.array([0, 0, 12]),
            ],
        )

        self.__assert_equality_on_point3s(
            align_translations_call_args[1],
            [
                np.array([0, 20.0001, 0]),
                np.array([0, 0, 12]),
            ],
        )

        compare_translations_call_args = compare_translations_mocked.call_args.args
        self.assertEqual(len(compare_translations_call_args[0]), 0)
        self.__assert_equality_on_point3s(
            compare_translations_call_args[1],
            [
                np.array([0, 20.0001, 0]),
                np.array([0, 0, 12]),
            ],
        )
        self.assertEqual(compare_translations_call_args[2], POINT3_RELATIVE_ERROR_THRESH)
        self.assertEqual(compare_translations_call_args[3], POINT3_ABS_ERROR_THRESH)

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


if __name__ == "__main__":
    unittest.main()
