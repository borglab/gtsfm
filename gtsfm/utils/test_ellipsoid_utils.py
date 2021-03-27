"""
Unit tests for functions in ellipsoid utils file.

Authors: Adi Singh
"""
import unittest
import numpy.testing as npt
import gtsfm.utils.ellipsoid as ellipsoid_utils
import numpy as np

class TestEllipsoidUtils(unittest.TestCase):
    def test_center_point_cloud_sample_points(self):
        """
        Testing the center_point_cloud() function with 5 sample points.
        """
        sample_points = [
            [6, 13, 9],
            [5, 9, 7],
            [6, 10, 9],
            [9, 13, 6],
            [5, 12, 9]
        ]
        computed = ellipsoid_utils.center_point_cloud(sample_points)
        expected = np.array([
            [-0.2,  1.6,  1 ],
            [-1.2, -2.4, -1 ],
            [-0.2, -1.4,  1 ],
            [ 2.8,  1.6, -2 ],
            [-1.2,  0.6,  1 ]
        ])
        npt.assert_almost_equal(computed, expected, 6)

    def test_center_point_cloud_wrong_dims(self):
        """
        Testing the center_point_cloud() function with 5 sample points of 2 dimensions.
        """
        sample_points = [
            [6, 13],
            [5, 9],
            [6, 10],
            [9, 13],
            [5, 12]
        ]
        self.assertRaises(TypeError, ellipsoid_utils.center_point_cloud, sample_points)

    def test_compute_convex_hull_sample_points(self):
        """
        Testing the compute_convex_hull() function with 5 sample points.
        """
        sample_points = np.array([
            [6, 13, 9],
            [5, 9, 7],
            [6, 10, 9],
            [9, 13, 6],
            [5, 12, 9]
        ])
        computed = ellipsoid_utils.compute_convex_hull(sample_points)
        computed = np.sort(computed, axis=0)

        expected = np.array([
            [ 5, 12,  9],
            [ 6, 13,  9],
            [ 6, 10,  9],
            [ 9, 13,  6]
        ])
        expected = np.sort(expected, axis=0)

        npt.assert_almost_equal(computed, expected, 6)

    def test_compute_convex_hull_wrong_dims(self):
        """
        Testing the compute_convex_hull() function with 5 sample points of 2 dimesions.
        """
        sample_points = np.array([
            [6, 13],
            [5, 9],
            [6, 10],
            [9, 13],
            [5, 12]
        ])
        self.assertRaises(TypeError, ellipsoid_utils.compute_convex_hull, sample_points)

    def test_fit_ls_ellipsoid_sample_points(self):
        """
        Testing the fit_ls_ellipsoid() function with 5 sample points.
        """
        sample_points = np.array([
            [0.1, 1.3, 0.9],
            [0.5, 0.9, 0.7],
            [0.6, 1.0, 0.9],
            [0.9, 1.3, 0.6],
            [0.5, 1.2, 0.9]
        ])
        computed = ellipsoid_utils.fit_ls_ellipsoid(sample_points)
        expected = np.array([0.125, 3.875, -10.5, -4, 10.304688, -1, -1, 0.25, 1.65625, -1])
        npt.assert_almost_equal(computed, expected, 6)

    def test_fit_ls_ellipsoid_wrong_dims(self):
        """
        Testing the fit_ls_ellipsoid() function with 5 sample points of dimension 2.
        """
        sample_points = np.array([
            [0.1, 1.3],
            [0.5, 0.9],
            [0.6, 1.0],
            [0.9, 1.3],
            [0.5, 1.2]
        ])
        self.assertRaises(TypeError, ellipsoid_utils.fit_ls_ellipsoid, sample_points)

    def test_extract_params_from_poly_sample_points(self):
        """
        Testing the extract_params_from_poly() function.
        """
        params = np.array([0.04026749, 0.04943968, 0.0519318 , 0.00137289, 0.01238955, 
                0.02562406, 0.03067298, -0.00346132, 0.02130539, -1])
        
        computed_center, computed_axes, computed_rot = ellipsoid_utils.extract_params_from_poly(params)

        expected_center = np.array([-0.35396952, 0.08774596, -0.18455235])
        expected_axes = np.array([3.94845524, 4.88893746, 5.37621732])
        expected_rot = np.array([[-0.20692656, -0.63630763, -0.74316485],
                    [-0.75280031, 0.58871614, -0.29445713],
                    [-0.62487846, -0.49852373, 0.60083358]])
        
        npt.assert_almost_equal(computed_center, expected_center, 6)
        npt.assert_almost_equal(computed_axes, expected_axes, 6)
        npt.assert_almost_equal(computed_rot, expected_rot, 6)

    def test_extract_params_from_poly_wrong_dims(self):
        """
        Testing the extract_params_from_poly() function with 11 parameters.
        """
        params = np.array([0.04026749, 0.04943968, 0.0519318 , 0.00137289, 0.01238955, 
                0.02562406, 0.03067298, -0.00346132, 0.02130539, -1, -1])
        self.assertRaises(TypeError, ellipsoid_utils.extract_params_from_poly, params)

    def test_apply_ellipsoid_rotation_sample_points(self):
        """
        Testing the apply_ellipsoid_rotation() function with 3 sample points and a matrix which rotates 180 degrees about z.
        """
        sample_rot = np.array([     #180 degrees rotation about z axis
            [-1,0,0],
            [0,-1,0],
            [0,0,1]
        ])
        sample_points = np.array([
            [2,3,5],
            [-1,4,6],
            [-2,-2,3]
        ])

        computed = ellipsoid_utils.apply_ellipsoid_rotation(sample_rot, sample_points)
        expected = [
            [-2,-3,5],
            [1,-4,6],
            [2,2,3]
        ]
        self.assertListEqual(computed, expected)

    def test_apply_ellipsoid_rotation_wrong_dims(self):
        """
        Testing the apply_ellipsoid_rotation() function with a rotation matrix of shape 3 x 2.
        """
        sample_rot = np.array([ 
            [-1,0],
            [0,-1],
            [0,0]
        ])
        sample_points = np.array([
            [2,3,5],
            [-1,4,6],
            [-2,-2,3]
        ])
        self.assertRaises(TypeError, ellipsoid_utils.apply_ellipsoid_rotation, sample_rot, sample_points)

if __name__ == "__main__":
    unittest.main()