"""Unit tests for verification utils.

Authors: Ayush Baid
"""
import unittest

import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Rot3, Unit3

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
            None, corr_i1.coordinates, corr_i2.coordinates, Cal3Bundler(), Cal3Bundler()
        )

        # compare the recovered R and U with the ground truth
        self.assertIsNone(i2Ri1)
        self.assertIsNone(i2Ui1)

    def test_compute_epipolar_distances(self):
        """Test for epipolar distance computation using 2 sets of points and the essential matrix."""
        points_i1 = np.array([[1.0, 3.5], [-2.0, 2.0]])
        points_i2 = np.array([[2.0, -1.0], [1.0, 0.0]])

        i2Ri1 = Rot3.RzRyRx(0, np.deg2rad(30), np.deg2rad(10))
        i2ti1 = np.array([-0.5, 2.0, 0])
        i2Ei1 = EssentialMatrix(i2Ri1, Unit3(i2ti1))

        expected = np.array([1.637, 1.850])

        computed = verification_utils.compute_epipolar_distances(points_i1, points_i2, i2Ei1.matrix())

        np.testing.assert_allclose(computed, expected, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
