"""Unit tests for utilities for MVS methods

Authors: Ren Liu, Ayush Baid
"""
import unittest

import numpy as np
from gtsam import PinholeCameraCal3Bundler, Pose3, Rot3

import gtsfm.densify.mvs_utils as mvs_utils


class TestMVSUtils(unittest.TestCase):
    """Unit tests for utilities for MVS methods."""

    def test_calculate_triangulation_angle_in_degrees(self) -> None:
        """Test the computation of triangulation angle using a simple example.
        Lengths of line segments are defined as follows:
                   5*sqrt(2)
                  X ---- C1
                  |    /
        5*sqrt(2) |  / 10 = 5*sqrt(2)*sqrt(2)
                  C2
        Cameras and point situated as follows in the x-z plane:
        (0,0,0)
             o---- +z
             |
             |
             +x
                      X (5,0,5)
        (10,0,0)
             o---- +z
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
        computed = mvs_utils.calculate_triangulation_angle_in_degrees(
            camera_1=PinholeCameraCal3Bundler(wT0),
            camera_2=PinholeCameraCal3Bundler(wT1),
            point_3d=point_3d,
        )
        self.assertAlmostEqual(computed, expected)

    def test_calculate_triangulation_angles_in_degrees(self) -> None:
        """Test the computation of triangulation angle using a simple example.
        Lengths of line segments are defined as follows:
                   5*sqrt(2)
                  X ---- C1
                  |    /
        5*sqrt(2) |  / 10 = 5*sqrt(2)*sqrt(2)
                  C2
        Cameras and point situated as follows in the x-z plane:
        (0,0,0)
             o---- +z
             |
             |
             +x
                      X (5,0,5)
        (10,0,0)
             o---- +z
             |
             |
             +x
        """
        camera_center_1 = np.array([0, 0, 0])
        camera_center_2 = np.array([10, 0, 0])
        points_3d = np.array([[5, 0, 5], [5, 0, 5], [5, 0, 5], [5, 0, 5]])

        expected = np.array([90, 90, 90, 90])

        wT0 = Pose3(Rot3(), camera_center_1)
        wT1 = Pose3(Rot3(), camera_center_2)

        computed = mvs_utils.calculate_triangulation_angles_in_degrees(
            camera_1=PinholeCameraCal3Bundler(wT0),
            camera_2=PinholeCameraCal3Bundler(wT1),
            points_3d=points_3d,
        )
        self.assertTrue(np.allclose(computed, expected))

    def test_piecewise_gaussian_below_expect_baseline_angle(self) -> None:
        """Unit test for the case that the angle between two coordinates is below the expect baseline angle,
        where sigma_1 is used to calculate the score"""

        score = mvs_utils.piecewise_gaussian(theta=4, theta_0=5, sigma_1=1, sigma_2=10)

        self.assertAlmostEqual(score, np.exp(-(1.0**2) / (2 * 1.0**2)))

    def test_piecewise_gaussian_above_expect_baseline_angle(self) -> None:
        """Unit test for the case that the angle between two coordinates is above the expect baseline angle,
        where sigma_2 is used to calculate the score"""

        score = mvs_utils.piecewise_gaussian(theta=10, theta_0=5, sigma_1=1, sigma_2=10)

        self.assertAlmostEqual(score, np.exp(-(5.0**2) / (2 * 10.0**2)))

    def test_cart_to_homogenous(self) -> None:
        """Test the cart_to_homogenous function correctly produces the homogenous coordinates"""

        n = 100
        uv = np.random.random([2, n])
        uv_homo = mvs_utils.cart_to_homogenous(uv)

        self.assertTrue(uv_homo.shape == (3, n))

    def test_estimate_minimum_voxel_size(self) -> None:
        """Test the estimate_minimum_voxel_size function correctly produces the minimum voxel size"""

        # ramdomly sample a normal-distributed point cloud with variances along each axis are 4, 1, 100
        mean = [1, 2, 3]
        # standard deviations are 2, 1, and 10
        cov = [[4, 0, 0], [0, 1, 0], [0, 0, 100]]
        points = np.random.multivariate_normal(mean, cov, 5000)

        scale = 0.01
        min_voxel_size = mvs_utils.estimate_minimum_voxel_size(points=points, scale=scale)
        self.assertAlmostEqual(min_voxel_size, 1 * scale, delta=0.1 * scale)

    def test_compute_downsampling_psnr(self) -> None:
        """Test the compute_downsampling_psnr function correctly produce the PSNR between two point clouds
        We use the dummy original and downsampled point clouds as following:
               o----o (1,1,1)
             / |   /|
            o--|--o |
            | o --|-o                                 (1, 1/2, 1/2)
            |/    |/                              o----o
            o-- --o                      (0, 1/2, 1/2)
        (0,0,0)
        original point cloud              downsampled point cloud

        The fitting ellipsoid's semi-axis lengths of the original point cloud is [0.5345, 0.5345, 0.5345]. Then estimate
        the diagonal of the point cloud's bounding box (dB) as the diagonal of the circumscribed rectangular
        parallelepiped of the ellipsoid, which is 1.852.

        For each point in the original point, the distances to the nearest neighbors in the downsampled point cloud
        (D_od) are all 0.5 * sqrt(2). And for each point in the downsampled point, the distances to the nearest n
        eighbors in the original point cloud (D_do) are all 0.5 * sqrt(2) as well.

        Then according to the formula: psnr = 20 log_10 (dB / max(RMS(D_od), RMS(D_do))), the psnr is approximately 8.36
        """

        original_point_cloud = np.array(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        )
        downsampled_point_cloud = np.array([[0, 0.5, 0.5], [1, 0.5, 0.5]])
        psnr = mvs_utils.compute_downsampling_psnr(original_point_cloud, downsampled_point_cloud)

        self.assertAlmostEqual(psnr, 8.36, 2)


if __name__ == "__main__":
    unittest.main()
