"""Tests for frontend's base verifier class.

Authors: Ayush Baid
"""

import pickle
import random
import unittest
from typing import Tuple

import dask
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix

from frontend.verifier.dummy_verifier import DummyVerifier


class TestVerifierBase(unittest.TestCase):
    """Unit tests for the Base Verifier class.

    Should be inherited by all verifier unit tests.
    """

    def setUp(self):
        super().setUp()

        self.verifier = DummyVerifier()

    def test_valid_verified_indices(self):
        """Test if valid indices in output."""

        # Repeat the experiment 10 times as we might not have successful
        # verification every time.

        for _ in range(10):
            _, verified_indices, keypoints_i1, keypoints_i2 = \
                self.__verify_random_inputs_with_exact_intrinsics()

            if verified_indices.size > 0:
                # check that the indices are not out of bounds
                self.assertTrue(np.all(verified_indices >= 0))
                self.assertTrue(
                    np.all(verified_indices[:, 0] < keypoints_i1.shape[0]))
                self.assertTrue(
                    np.all(verified_indices[:, 1] < keypoints_i2.shape[0]))
            else:
                # we have a meaningless test
                self.assertTrue(True)

    def test_verify_empty_matches(self):
        """Tests the output when there are no match indices."""

        keypoints_i1 = generate_random_keypoints(10, [250, 300])
        keypoints_i2 = generate_random_keypoints(12, [400, 300])
        match_indices = np.array([], dtype=np.int32)
        intrinsics_i1 = Cal3Bundler()
        intrinsics_i2 = Cal3Bundler()

        i2Ei1, verified_indices = self.verifier.verify_with_exact_intrinsics(
            keypoints_i1, keypoints_i2, match_indices, intrinsics_i1, intrinsics_i2
        )

        self.assertIsNone(i2Ei1)
        self.assertEqual(0, verified_indices.size)

    def test_create_computation_graph(self):
        """Checks that the dask computation graph produces the same results as
        direct APIs."""

        # Set up 3 pairs of inputs to the verifier
        num_images = 6
        image_indices = [(0, 1), (4, 3), (2, 5)]

        # creating inputs for verification and use GTSFM's direct API to get
        # expected results
        keypoints_list = [None]*num_images
        matches_dict = dict()
        intrinsics_list = [None]*num_images

        expected_results = dict()
        for (i1, i2) in image_indices:
            keypoints_i1, keypoints_i2, matches_i1i2, \
                intrinsics_i1, intrinsics_i2 = \
                generate_random_input_for_verifier()

            keypoints_list[i1] = keypoints_i1
            keypoints_list[i2] = keypoints_i2

            matches_dict[(i1, i2)] = matches_i1i2

            intrinsics_list[i1] = intrinsics_i1
            intrinsics_list[i2] = intrinsics_i2

            verification_result_i1i2 = \
                self.verifier.verify_with_exact_intrinsics(
                    keypoints_i1,
                    keypoints_i2,
                    matches_i1i2,
                    intrinsics_i1,
                    intrinsics_i2
                )

            expected_results[(i1, i2)] = verification_result_i1i2

        # Convert the inputs to computation graphs
        detection_graph = [dask.delayed(x) for x in keypoints_list]
        matcher_graph = {image_indices: dask.delayed(match) for
                         (image_indices, match) in matches_dict.items()}
        intrinsics_graph = [dask.delayed(x) for x in intrinsics_list]

        # generate the computation graph for the verifier
        computation_graph = self.verifier.create_computation_graph(
            detection_graph,
            matcher_graph,
            intrinsics_graph,
            exact_intrinsics_flag=True
        )

        with dask.config.set(scheduler='single-threaded'):
            dask_results = dask.compute(computation_graph)[0]

        # compare the lengths of two results dictionaries
        self.assertEqual(len(expected_results), len(dask_results))

        # compare the values in two dictionaries
        for indices_i1i2 in dask_results.keys():
            i2Ei1_dask, verified_indices_i1i2_dask = dask_results[indices_i1i2]

            i2Ei1_expected, verified_indices_i1i2_expected = \
                expected_results[indices_i1i2]

            if i2Ei1_expected is None:
                self.assertIsNone(i2Ei1_dask)
            else:
                self.assertTrue(i2Ei1_expected.equals(i2Ei1_dask, 1e-2))

            np.testing.assert_array_equal(
                verified_indices_i1i2_expected, verified_indices_i1i2_dask)

    def test_pickleable(self):
        """Tests that the verifier object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.verifier)
        except TypeError:
            self.fail("Cannot dump verifier using pickle")

    def __verify_random_inputs_with_exact_intrinsics(self) -> \
            Tuple[EssentialMatrix, np.ndarray, np.ndarray, np.ndarray]:
        """Generates random inputs for pair (#i1, #i2) and peform verification
        by treating intrinsics as exact.

        Returns:
            Computed essential matrix i2Ei1.
            Indices of keypoints which are verified correspondences.
            Keypoints from #i1 which were input to the verifier.
            Keypoints from #i2 which were input to the verifier.
        """

        keypoints_i1, keypoints_i2, match_indices, \
            intrinsics_i1, intrinsics_i2 = \
            generate_random_input_for_verifier()

        i2Ei1, verified_indices = self.verifier.verify_with_exact_intrinsics(
            keypoints_i1,
            keypoints_i2,
            match_indices,
            intrinsics_i1,
            intrinsics_i2
        )

        return i2Ei1, verified_indices, keypoints_i1, keypoints_i2


def generate_random_keypoints(num_keypoints: int,
                              image_shape: Tuple[int, int]) -> np.ndarray:
    """Generates random features within the image bounds.

    Args:
        num_keypoints: number of features to generate.
        image_shape: size of the image.

    Returns:
        generated features.
    """

    if num_keypoints == 0:
        return np.array([])

    return np.random.randint(
        [0, 0], high=image_shape, size=(num_keypoints, 2)
    ).astype(np.float32)


def generate_random_input_for_verifier() -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, Cal3Bundler, Cal3Bundler]:
    """Generattes random inputs for verification

    Returns:
        Keypoints for image #i1.
        Keypoints for image #i2.
        Indices of match between image pair (#i1, #i2).
        Intrinsics for image #i1.
        Intrinsics for image #i2.
    """

    # Randomly generate number of keypoints
    num_keypoints_i1 = random.randint(0, 100)
    num_keypoints_i2 = random.randint(0, 100)

    # randomly generate image shapes
    image_shape_i1 = [random.randint(100, 400), random.randint(100, 400)]
    image_shape_i2 = [random.randint(100, 400), random.randint(100, 400)]

    # generate intrinsics from image shapes
    intrinsics_i1 = Cal3Bundler(
        fx=min(image_shape_i1[0], image_shape_i1[1]),
        k1=0,
        k2=0,
        u0=image_shape_i1[0]/2,
        v0=image_shape_i1[1]/2,
    )

    intrinsics_i2 = Cal3Bundler(
        fx=min(image_shape_i2[0], image_shape_i2[1]),
        k1=0,
        k2=0,
        u0=image_shape_i2[0]/2,
        v0=image_shape_i2[1]/2,
    )

    # randomly generate the keypoints
    keypoints_i1 = generate_random_keypoints(
        num_keypoints_i1, image_shape_i1)
    keypoints_i2 = generate_random_keypoints(
        num_keypoints_i2, image_shape_i2)

    # randomly generate matches
    num_matches = random.randint(0, min(num_keypoints_i1, num_keypoints_i2))
    if num_matches == 0:
        matching_indices_i1i2 = np.array([], dtype=np.int32)
    else:
        matching_indices_i1i2 = np.empty((num_matches, 2), dtype=np.int32)
        matching_indices_i1i2[:, 0] = np.random.choice(
            num_keypoints_i1, size=(num_matches,), replace=False)
        matching_indices_i1i2[:, 1] = np.random.choice(
            num_keypoints_i2, size=(num_matches,), replace=False)

    return keypoints_i1, keypoints_i2, matching_indices_i1i2, \
        intrinsics_i1, intrinsics_i2


if __name__ == "__main__":
    unittest.main()
