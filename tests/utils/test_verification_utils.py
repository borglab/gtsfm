"""Unit tests for verification utils.

Authors: Ayush Baid
"""
import unittest

import numpy as np
from gtsam import Cal3Bundler

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


if __name__ == "__main__":
    unittest.main()
