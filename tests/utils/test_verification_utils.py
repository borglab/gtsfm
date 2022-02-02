"""Unit tests for verification utils.

Authors: Ayush Baid
"""
import unittest

import numpy as np
from gtsam import Cal3Bundler, Pose3, Rot3

import gtsfm.utils.verification as verification_utils
from tests.frontend.verifier.test_verifier_base import simulate_two_planes_scene


class TestVerificationUtils(unittest.TestCase):
    """Class containing unit tests for verification utils."""

    def test_recover_relative_pose_from_essential_matrix_valid(self):
        """Test for function to extract relative pose from essential matrix."""

        # simulate correspondences and the essential matrix
        corr_i1, corr_i2, i2Ei1 = simulate_two_planes_scene(10, 10)

        i2Ri1, i2Ui1 = verification_utils.recover_relative_pose_from_essential_matrix(
            i2Ei1.matrix(), corr_i1.coordinates, corr_i2.coordinates, Cal3Bundler(), Cal3Bundler()
        )

        # compare the recovered R and U with the ground truth
        self.assertTrue(i2Ri1.equals(i2Ei1.rotation(), 1e-3))
        self.assertTrue(i2Ui1.equals(i2Ei1.direction(), 1e-3))

    def test_recover_relative_pose_from_essential_matrix_none(self):
        """Test for function to extract relative pose from essential matrix."""

        # simulate correspondences and the essential matrix
        corr_i1, corr_i2, _ = simulate_two_planes_scene(10, 10)

        i2Ri1, i2Ui1 = verification_utils.recover_relative_pose_from_essential_matrix(
            i2Ei1=None,
            verified_coordinates_i1=corr_i1.coordinates,
            verified_coordinates_i2=corr_i2.coordinates,
            camera_intrinsics_i1=Cal3Bundler(),
            camera_intrinsics_i2=Cal3Bundler(),
        )

        # compare the recovered R and U with the ground truth
        self.assertIsNone(i2Ri1)
        self.assertIsNone(i2Ui1)

    def test_compute_epipolar_distances_sed(self):
        """Test for epipolar distance computation using 2 sets of points and the fundamental matrix."""

        #####
        # Test 1: testing on a simple simulated case
        #####
        points_i1 = np.array([[1.0, 3.5], [-2.0, 2.0]])
        points_i2 = np.array([[2.0, -1.0], [1.0, 0.0]])
        i2Fi1 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        # i2Fi1 @ x1 = [4.5, 1.0, 1.0] and [3.0, -2.0, -2.0]. Norms^2 = 21.25 and 13.0
        # x2.T @ i2Fi1 = [0.0, 2.0, 2.0] and [1.0, 1.0, 1.0]. Norms^2 = 4.0 and 2.0
        # point line dot product: 9 and 1
        expected = np.array([81 * (1 / 21.25 + 1 / 4.0), 1 * (1 / 13.0 + 1 / 2.0)])
        computed = verification_utils.compute_epipolar_distances_sq_sed(points_i1, points_i2, i2Fi1)
        np.testing.assert_allclose(computed, expected, rtol=1e-3)

        #####
        # Test 2: testing on an example from real world
        #####
        i2Fi1 = np.array(
            [
                [7.41572822e-09, 4.26005557e-07, -2.61114657e-04],
                [-4.92270651e-07, 4.29568438e-09, 6.95083578e-04],
                [2.89444929e-04, -1.49345006e-05, -4.01395060e-01],
            ]
        )
        points_i1 = np.array([[1553, 622], [1553, 622]])
        points_i2 = np.array([[357, 662], [818, 517]])
        expected = np.array([4.483719e00, 6.336384e04])
        computed = verification_utils.compute_epipolar_distances_sq_sed(points_i1, points_i2, i2Fi1)
        np.testing.assert_allclose(computed, expected, rtol=1e-3)

    def test_compute_epipolar_distances_sampson(self):
        """Test for epipolar distance computation using 2 sets of points and the fundamental matrix."""

        #####
        # Test 1: testing on a simple simulated case
        #####
        points_i1 = np.array([[1.0, 3.5], [-2.0, 2.0]])
        points_i2 = np.array([[2.0, -1.0], [1.0, 0.0]])
        i2Fi1 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        # i2Fi1 @ x1 = [4.5, 1.0, 1.0] and [3.0, -2.0, -2.0]. Norms^2 = 21.25 and 13.0
        # x2.T @ i2Fi1 = [0.0, 2.0, 2.0] and [1.0, 1.0, 1.0]. Norms^2 = 4.0 and 2.0
        # point line dot product: 9 and 1
        expected = np.array([81 / (21.25 + 4.0), 1 / (13.0 + 2.0)])
        computed = verification_utils.compute_epipolar_distances_sq_sampson(points_i1, points_i2, i2Fi1)
        np.testing.assert_allclose(computed, expected, rtol=1e-3)

        #####
        # Test 2: testing on an example from real world (argoverse)
        #####
        i2Fi1 = np.array(
            [
                [7.41572822e-09, 4.26005557e-07, -2.61114657e-04],
                [-4.92270651e-07, 4.29568438e-09, 6.95083578e-04],
                [2.89444929e-04, -1.49345006e-05, -4.01395060e-01],
            ]
        )
        points_i1 = np.array([[1553, 622], [1553, 622]])
        points_i2 = np.array([[357, 662], [818, 517]])
        expected = np.array([6.744895e-01, 2.397196e03])
        computed = verification_utils.compute_epipolar_distances_sq_sampson(points_i1, points_i2, i2Fi1)
        np.testing.assert_allclose(computed, expected, rtol=1e-3)

    def test_recover_pose_from_projection_matrix_palace(self) -> None:
        """Ensure we can recover camera pose and intrinsics from GT camera projection matrix.

        Example taken from Carl Olsson's Palace of Fine Arts (281 image) dataset, from view 0,
        for image w/ shape (H,W) = (1296,1936).
        """
        M = np.array(
            [
                [2.28437969e03, -1.78828636e02, -1.16188350e03, 0.00000000e00],
                [6.51503486e02, 2.35577922e03, 4.14170147e02, 0.00000000e00],
                [7.42992540e-01, -5.60441538e-02, 6.66949127e-01, 0.00000000e00],
            ]
        )
        K, wTc = verification_utils.decompose_camera_projection_matrix(M)

        # fmt: off
        K_expected = np.array(
            [
                [2394, 0, 932],
                [0, 2398, 628],
                [0, 0, 1]
            ]
        )
        # fmt: on
        assert np.allclose(K, K_expected, atol=1.0)

        wTc_expected = Pose3(
            r=Rot3(
                np.array(
                    [
                        [0.664853, 0.0770217, 0.742993],
                        [-0.0528724, 0.997027, -0.0560442],
                        [-0.745101, -0.00202269, 0.666949],
                    ]
                )
            ),
            t=np.array([0, 0, 0]),
        )
        assert wTc.equals(wTc_expected, tol=1e-5)

    def test_recover_pose_from_projection_matrix_door(self) -> None:
        """Ensure we can recover camera pose and intrinsics from GT camera projection matrix.

        Note: taken taken from 0th and 1st frames of the Door dataset.
        """

        M0 = np.array(
            [
                [2.39811854e03, 0.00000000e00, 6.28264995e02, 0.00000000e00],
                [0.00000000e00, 2.39395217e03, 9.32382177e02, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 1.00000000e00, 0.00000000e00],
            ]
        )
        K0, wT0 = verification_utils.decompose_camera_projection_matrix(M0)

        M1 = np.array(
            [
                [2.43113498e03, -3.64724612e01, 4.83675431e02, 1.95488302e03],
                [9.38533874e01, 2.39686078e03, 9.20105125e02, -1.50650797e01],
                [5.98843866e-02, 3.46851131e-03, 9.98199294e-01, -6.16173247e-02],
            ]
        )
        K1, wT1 = verification_utils.decompose_camera_projection_matrix(M1)

        # fmt: off
        K_expected = np.array(
            [
                [2398, 0, 628],
                [0, 2394, 932],
                [0, 0, 1]
            ]
        )
        # fmt: on
        assert np.allclose(K0, K_expected, atol=1.0)
        # GT intrinsics should match between view 0 and view 1
        assert np.allclose(K0, K1, atol=1e-7)

        # first pose is the origin for the world frame.
        assert Pose3().equals(wT0, tol=1e-12)
        assert not Pose3().equals(wT1, tol=0.01)


if __name__ == "__main__":
    unittest.main()
