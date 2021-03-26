"""Tests for frontend's joint matcher-verifier class.

Authors: Ayush Baid
"""
import pickle
import random
import unittest
from typing import Tuple

import dask
import numpy as np
from gtsam import Cal3Bundler

import gtsfm.utils.features as feature_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.dummy_matcher import DummyMatcher
from gtsfm.frontend.verifier.dummy_verifier import DummyVerifier
from gtsfm.frontend.matcher_verifier.combination_matcher_verifier import CombinationMatcherVerifier


class TestMatcherVerifierBase(unittest.TestCase):
    """Unit tests for MatcherVerifierBase.

    Should be inherited by all matcher-verifier unit tests.
    """

    def setUp(self):
        super().setUp()

        self.matcher_verifier = CombinationMatcherVerifier(DummyMatcher(), DummyVerifier())

    def test_create_computation_graph(self):
        """Checks that the dask computation graph produces the same results as direct APIs."""

        # creating inputs for verification
        (
            keypoints_i1,
            keypoints_i2,
            descriptors_i1,
            descriptors_i2,
            intrinsics_i1,
            intrinsics_i2,
        ) = generate_random_inputs()

        for use_intrinsics_in_verification in (True, False):
            # and use GTSFM's direct API to get expected results
            expected_results = self.matcher_verifier.match_and_verify(
                keypoints_i1,
                keypoints_i2,
                descriptors_i1,
                descriptors_i2,
                intrinsics_i1,
                intrinsics_i2,
                use_intrinsics_in_verification,
            )
            expected_i2Ri1 = expected_results[0]
            expected_i2Ui1 = expected_results[1]
            expected_v_corr_idxs = expected_results[2]

            # generate the computation graph for the verifier
            (delayed_i2Ri1, delayed_i2Ui1, delayed_v_corr_idxs,) = self.matcher_verifier.create_computation_graph(
                dask.delayed(keypoints_i1),
                dask.delayed(keypoints_i2),
                dask.delayed(descriptors_i1),
                dask.delayed(descriptors_i2),
                dask.delayed(intrinsics_i1),
                dask.delayed(intrinsics_i2),
                use_intrinsics_in_verification,
            )

            with dask.config.set(scheduler="single-threaded"):
                i2Ri1, i2Ui1, v_corr_idxs = dask.compute(delayed_i2Ri1, delayed_i2Ui1, delayed_v_corr_idxs)

            if expected_i2Ri1 is None:
                self.assertIsNone(i2Ri1)
            else:
                self.assertTrue(expected_i2Ri1.equals(i2Ri1, 1e-2))
            if expected_i2Ui1 is None:
                self.assertIsNone(i2Ui1)
            else:
                self.assertTrue(expected_i2Ui1.equals(i2Ui1, 1e-2))
            np.testing.assert_array_equal(v_corr_idxs, expected_v_corr_idxs)

    def test_pickleable(self):
        """Tests that the verifier object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.matcher_verifier)
        except TypeError:
            self.fail("Cannot dump matcher-verifier using pickle")


def generate_random_inputs() -> Tuple[Keypoints, Keypoints, np.ndarray, np.ndarray, Cal3Bundler, Cal3Bundler]:
    """Generates random inputs for verification.

    Returns:
        Keypoints for image #i1.
        Keypoints for image #i2.
        Descriptors for image #i1.
        Descriptors for image #i2.
        Intrinsics for image #i1.
        Intrinsics for image #i2.
    """

    # Randomly generate number of keypoints
    num_keypoints_i1 = random.randint(0, 100)
    num_keypoints_i2 = random.randint(0, 100)

    # randomly generate image shapes
    image_shape_i1 = (random.randint(100, 400), random.randint(100, 400))
    image_shape_i2 = (random.randint(100, 400), random.randint(100, 400))

    # generate intrinsics from image shapes
    intrinsics_i1 = Cal3Bundler(
        fx=min(image_shape_i1[0], image_shape_i1[1]), k1=0, k2=0, u0=image_shape_i1[0] / 2, v0=image_shape_i1[1] / 2,
    )

    intrinsics_i2 = Cal3Bundler(
        fx=min(image_shape_i2[0], image_shape_i2[1]), k1=0, k2=0, u0=image_shape_i2[0] / 2, v0=image_shape_i2[1] / 2,
    )

    # randomly generate the keypoints
    keypoints_i1 = feature_utils.generate_random_keypoints(num_keypoints_i1, image_shape_i1)
    keypoints_i2 = feature_utils.generate_random_keypoints(num_keypoints_i2, image_shape_i2)

    # generate their descriptors
    descriptor_dim = 256
    descriptors_i1 = np.random.randn(len(keypoints_i1), descriptor_dim)
    descriptors_i2 = np.random.randn(len(keypoints_i2), descriptor_dim)

    return (
        keypoints_i1,
        keypoints_i2,
        descriptors_i1,
        descriptors_i2,
        intrinsics_i1,
        intrinsics_i2,
    )


if __name__ == "__main__":
    unittest.main()
