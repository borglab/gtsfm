"""Unit tests for verification utils.

Authors: Ayush Baid
"""
import unittest

from gtsam import Cal3Bundler

import gtsfm.utils.verification as verification_utils
from tests.frontend.verifier.test_verifier_base import simulate_two_planes_scene


class TestVerificationUtils(unittest.TestCase):
    """Class containing unit tests for verification utils."""

    def test_recover_relative_pose_from_essential_matrix(self):
        """Test for function to extract relative pose from essential matrix."""

        # simulate correspondences and the essential matrix
        corr_i1, corr_i2, i2Ei1 = simulate_two_planes_scene(10, 10)

        i2Ri1, i2Ui1 = \
            verification_utils.recover_relative_pose_from_essential_matrix(
                i2Ei1.matrix(),
                corr_i1.coordinates,
                corr_i2.coordinates,
                Cal3Bundler(),
                Cal3Bundler()
            )

        # compare the recovered R and U with the ground truth
        self.assertTrue(i2Ri1.equals(i2Ei1.rotation(), 1e-3))
        self.assertTrue(i2Ui1.equals(i2Ei1.direction(), 1e-3))
