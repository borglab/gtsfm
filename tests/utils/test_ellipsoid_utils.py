"""Unit tests for functions in ellipsoid utils file.

Authors: Adi Singh
"""
import unittest

import numpy as np
import numpy.testing as npt

import gtsfm.utils.ellipsoid as ellipsoid_utils


class TestEllipsoidUtils(unittest.TestCase):
    """Class containing all unit tests for ellipsoid utils."""

    def test_center_point_cloud(self) -> None:
        """Tests the center_point_cloud() function with 3 sample points.

        Means of x,y,z is clearly (2,2,2), so centering the point cloud yields:
            (1,1,1) -> (-1,-1,-1)
            (2,2,2) -> (0,0,0)
            (3,3,3) -> (1,1,1)
        """

        sample_points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        computed, mean = ellipsoid_utils.center_point_cloud(sample_points)
        expected = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        npt.assert_almost_equal(computed, expected, decimal=6)

    def test_center_point_cloud_wrong_dims(self) -> None:
        """Tests the center_point_cloud() function with 5 sample points of 2 dimensions."""

        sample_points = np.array([[6, 13], [5, 9], [6, 10]])
        self.assertRaises(TypeError, ellipsoid_utils.center_point_cloud, sample_points)

    def test_remove_outlier_points(self) -> None:
        """Tests the remove_outlier_points() function with 5 sample points."""

        # fmt: off
        sample_points = np.array(
            [
                [0.5, 0.6, 0.8],
                [0.9, 1, 0.2],
                [20, 20, 20],  # the outlier point to be filtered
                [0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3],
            ]
        )
        # fmt: on

        computed = ellipsoid_utils.remove_outlier_points(sample_points)
        expected = np.array([[0.5, 0.6, 0.8], [0.9, 1, 0.2], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
        npt.assert_almost_equal(computed, expected, decimal=6)

    def test_remove_outlier_points_wrong_dims(self) -> None:
        """Tests the remove_outlier_points() function with 5 sample points of 2 dimensions."""

        sample_points = np.array([[6, 13], [5, 9], [6, 10], [9, 13], [5, 12]])
        self.assertRaises(TypeError, ellipsoid_utils.remove_outlier_points, sample_points)

    def test_get_alignment_rotation_matrix_from_svd(self) -> None:
        """Tests the get_alignment_rotation_matrix_from_svd() function with 6 sample points. Transforms a rotated cross
        to points lying on the x and y axes.

        sample_points:        aligned_points:

             |                       |
         o   |                       o
           o | o                     |
        -------------   ==>  --o--o-----o--o--
           o | o                     |
             |   o                   o
             |                       |

         o = point
        """

        # fmt: off
        sample_points = np.array(
            [
                [1, 1, 0], 
                [-1, 1, 0], 
                [-2, 2, 0], 
                [-1, -1, 0], 
                [1, -1, 0], 
                [2, -2, 0]
            ]
        )
        # fmt: on

        computed_rotation = ellipsoid_utils.get_alignment_rotation_matrix_from_svd(sample_points)
        num = np.sqrt(2) / 2
        expected_rotation = np.array([[-num, num, 0], [-num, -num, 0], [0, 0, 1]])
        npt.assert_almost_equal(computed_rotation, expected_rotation, decimal=6)

        # Apply the rotation transformation to sample_points
        # Verify that every aligned point's x or y coordinate is 0
        aligned_points = sample_points @ computed_rotation.T

        closeToZero = np.isclose(aligned_points[:, :2], 0)
        assert np.all(closeToZero[:, 0] | closeToZero[:, 1])

    def test_get_alignment_rotation_matrix_from_svd_wrong_dims(self) -> None:
        """Tests the get_alignment_rotation_matrix_from_svd() function with 6 sample points of 2 dimensions."""

        sample_points = np.array([[1, 1], [-1, 1], [-2, 2], [-1, -1], [1, -1], [2, -2]])
        self.assertRaises(TypeError, ellipsoid_utils.get_alignment_rotation_matrix_from_svd, sample_points)


if __name__ == "__main__":
    unittest.main()
