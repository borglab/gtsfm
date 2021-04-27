"""Unit tests for functions in ellipsoid utils file.

Authors: Adi Singh
"""
import gtsfm.utils.ellipsoid as ellipsoid_utils
import numpy as np
import numpy.testing as npt
import unittest


class TestEllipsoidUtils(unittest.TestCase):
    """Class containing all unit tests for ellipsoid utils."""

    def test_center_point_cloud(self):
        """Tests the center_point_cloud() function with 5 sample points."""

        sample_points = np.array([[6, 13, 9], [5, 9, 7], [6, 10, 9], [9, 13, 6], [5, 12, 9]])
        computed, means = ellipsoid_utils.center_point_cloud(sample_points)
        expected = np.array([[-0.2, 1.6, 1], [-1.2, -2.4, -1], [-0.2, -1.4, 1], [2.8, 1.6, -2], [-1.2, 0.6, 1]])
        npt.assert_almost_equal(computed, expected, 6)

    def test_center_point_cloud_wrong_dims(self):
        """Tests the center_point_cloud() function with 5 sample points of 2 dimensions."""

        sample_points = np.array([[6, 13], [5, 9], [6, 10], [9, 13], [5, 12]])
        self.assertRaises(TypeError, ellipsoid_utils.center_point_cloud, sample_points)

    def test_filter_outlier_points(self):
        """Tests the filter_outlier_points() function with 5 sample points."""

        sample_points = np.array(
            [
                [0.5, 0.6, 0.8],
                [0.9, 1, 0.2],
                [20, 20, 20],  # the outlier point to be filtered
                [0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3],
            ]
        )
        computed = ellipsoid_utils.filter_outlier_points(sample_points)
        expected = np.array([[0.5, 0.6, 0.8], [0.9, 1, 0.2], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
        npt.assert_almost_equal(computed, expected, 6)

    def test_filter_outlier_points_wrong_dims(self):
        """Tests the filter_point_cloud() function with 5 sample points of 2 dimensions."""

        sample_points = np.array([[6, 13], [5, 9], [6, 10], [9, 13], [5, 12]])
        self.assertRaises(TypeError, ellipsoid_utils.filter_outlier_points, sample_points)

    def test_get_rotation_matrix(self):
        """Tests the get_rotation_matrix() function with 5 sample points."""

        sample_points = np.array([[1, 1, 0], [2, 2, 0], [-1, -1, 0], [-2, -2, 0], [-1, 1, 0], [1, -1, 0]])
        computed = ellipsoid_utils.get_rotation_matrix(sample_points)

        sqrt2_2 = np.sqrt(2) / 2
        expected = np.array([[-1 * sqrt2_2, -1 * sqrt2_2, 0], [sqrt2_2, -1 * sqrt2_2, 0], [0, 0, 1]])
        npt.assert_almost_equal(computed, expected, 6)

    def test_get_rotation_matrix_wrong_dims(self):
        """Tests the get_rotation_matrix() function with sample points of 2 dimensions."""

        sample_points = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2], [-1, 1], [1, -1]])
        self.assertRaises(TypeError, ellipsoid_utils.get_rotation_matrix, sample_points)

    def test_apply_ellipsoid_rotation(self):
        """Tests the apply_ellipsoid_rotation() function with sample points."""

        sample_rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        sample_points = np.array([[1, 1, 0], [2, 2, 0], [-1, -1, 0], [-2, -2, 0], [-1, 1, 0], [1, -1, 0]])
        computed = ellipsoid_utils.apply_ellipsoid_rotation(sample_rot, sample_points)
        expected = np.array([[0, -1, 1], [0, -2, 2], [0, 1, -1], [0, 2, -2], [0, -1, -1], [0, 1, 1]])
        npt.assert_almost_equal(computed, expected, 6)

    def test_apply_ellipsoid_rotation_wrong_dims(self):
        """Tests the apply_ellipsoid_rotation() function with sample points of 2 dimensions."""

        sample_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        sample_points = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2], [-1, 1], [1, -1]])
        self.assertRaises(TypeError, ellipsoid_utils.get_rotation_matrix, sample_rot, sample_points)


if __name__ == "__main__":
    unittest.main()
