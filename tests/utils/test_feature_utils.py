"""Unit test for common feature utils."""
import unittest

import numpy as np

from gtsam import Cal3Bundler, EssentialMatrix, Rot3, Unit3

import gtsfm.utils.features as feature_utils


class TestFeatureUtils(unittest.TestCase):
    """Class containing all unit tests for feature utils."""

    def test_normalize_coordinates(self):
        coordinates = np.array([[10.0, 20.0], [25.0, 12.0], [30.0, 33.0]])

        intrinsics = Cal3Bundler(fx=100, k1=0.0, k2=0.0, u0=20.0, v0=30.0)

        normalized_coordinates = feature_utils.normalize_coordinates(coordinates, intrinsics)

        expected_coordinates = np.array([[-0.1, -0.1], [0.05, -0.18], [0.1, 0.03]])

        np.testing.assert_allclose(normalized_coordinates, expected_coordinates)

    def test_convert_to_homogenous_coordinates_on_valid_input(self):
        """Tests conversion to homogenous coordinates on valid input."""

        non_homogenous_coordinates = np.random.rand(10, 2)
        computed = feature_utils.convert_to_homogenous_coordinates(non_homogenous_coordinates)

        np.testing.assert_allclose(computed[:, :2], non_homogenous_coordinates)
        np.testing.assert_allclose(computed[:, 2], 1)

    def test_convert_to_homogenous_coordinates_on_empty_input(self):
        """Tests conversion to homogenous coordinates on empty input."""

        non_homogenous_coordinates = np.array([])
        computed = feature_utils.convert_to_homogenous_coordinates(non_homogenous_coordinates)

        self.assertEqual(computed.size, 0)

    def test_convert_to_homogenous_coordinates_on_none_input(self):
        """Tests conversion to homogenous coordinates on None input."""

        non_homogenous_coordinates = None
        computed = feature_utils.convert_to_homogenous_coordinates(non_homogenous_coordinates)

        self.assertIsNone(computed)

    def test_convert_to_homogenous_coordinates_incorrect_dim(self):
        """Tests conversion to homogenous coordinates on input which is already
        3 dimensional."""

        # test on invalid input
        non_homogenous_coordinates = np.random.rand(5, 3)

        self.assertRaises(
            TypeError,
            feature_utils.convert_to_homogenous_coordinates,
            non_homogenous_coordinates,
        )

    def test_convert_to_epipolar_lines_valid_input(self):
        """Test conversion of valid 2D points to epipolar lines using the essential matrix."""

        points = np.array(
            [
                [10.0, -5.0],
                [3.5, 20.0],
            ]
        )  # 2d points in homogenous coordinates
        essential_mat = EssentialMatrix(Rot3.RzRyRx(0, np.deg2rad(45), 0), Unit3(np.array([-5, 2, 0])))

        computed = feature_utils.convert_to_epipolar_lines(points, essential_mat)

        expected = np.array(
            [
                [-2.36351579, -5.90878948, 1.75364193],
                [-0.65653216, -1.64133041, -19.75129171],
            ]
        )

        np.testing.assert_allclose(computed, expected)

    def test_convert_to_epipolar_lines_empty_input(self):
        """Test conversion of 0 2D points to epipolar lines using the essential matrix."""

        points = np.array([])  # 2d points in homogenous coordinates
        essential_mat = EssentialMatrix(Rot3.RzRyRx(0, np.deg2rad(45), 0), Unit3(np.array([-5, 2, 0])))

        computed = feature_utils.convert_to_epipolar_lines(points, essential_mat)

        self.assertEqual(computed.size, 0)

    def test_convert_to_epipolar_lines_none_input(self):
        """Test conversion of None to epipolar lines using the essential matrix."""

        points = None
        essential_mat = EssentialMatrix(Rot3.RzRyRx(0, np.deg2rad(45), 0), Unit3(np.array([-5, 2, 0])))

        computed = feature_utils.convert_to_epipolar_lines(points, essential_mat)

        self.assertIsNone(computed)

    def test_compute_point_line_distances(self):
        """Test for 2D point-line distance computation."""

        points = np.array(
            [
                [-2.0, 1.0],
                [5.0, 1.0],
                [2.0, 3.0],
            ]
        )
        lines = np.array(  # coefficients (a, b, c) for the line ax + by + cz = 0
            [
                [4.0, -3.0, -4.0],
                [0.0, -0.5, 1.0],
                [-3.0, -4.0, -5.0],
            ]
        )
        expected = np.array([3.0, 1.0, 4.6])
        computed = feature_utils.compute_point_line_distances(points, lines)

        np.testing.assert_allclose(computed, expected)


if __name__ == "__main__":
    unittest.main()