"""Tests for frontend's base matcher class.

Authors: Ayush Baid
"""

import pickle
import random
import unittest
from typing import Tuple

import dask
import numpy as np

from gtsfm.frontend.matcher.dummy_matcher import DummyMatcher


class TestMatcherBase(unittest.TestCase):
    """Unit tests for MatcherBase.

    Should be inherited by all matcher unit tests.
    """

    def setUp(self):
        super().setUp()

        self.matcher = DummyMatcher()

    def test_match_valid_indices(self):
        """Tests if matched indices are valid feature indices."""

        # run matching on a random input pair
        (
            result,
            descriptors_im1,
            descriptors_im2,
        ) = self.__generate_matches_on_random_descriptors()

        if result.size:
            # check that the match indices are out of bounds
            self.assertTrue(np.all((result[:, 0] >= 0) & (result[:, 0] < descriptors_im1.shape[0])))
            self.assertTrue(np.all((result[:, 1] >= 0) & (result[:, 1] < descriptors_im2.shape[0])))

    def test_empty_input(self):
        """Tests the matches when there are no descriptors."""

        num_descriptors = random.randint(5, 15)

        descriptor_dim = random.randint(2, 10)  # dimensionality

        descriptors = generate_random_binary_descriptors(num_descriptors, descriptor_dim)

        # the first descriptor is empty
        result = self.matcher.match(generate_random_binary_descriptors(0, descriptor_dim), descriptors)

        self.assertEqual(0, result.size)

        # the second descriptor is empty
        result = self.matcher.match(descriptors, generate_random_binary_descriptors(0, descriptor_dim))

        self.assertEqual(0, result.size)

        # both descriptors are empty
        result = self.matcher.match(
            generate_random_binary_descriptors(0, descriptor_dim),
            generate_random_binary_descriptors(0, descriptor_dim),
        )

        self.assertEqual(0, result.size)

    def test_one_to_one_constraint(self):
        """Tests that each index from an image is used atmost once."""

        # get a result
        result, _, _ = self.__generate_matches_on_random_descriptors()

        # form sets from the two index columns
        set_index_im1 = set(result[:, 0].tolist())
        set_index_im2 = set(result[:, 1].tolist())

        # if we have 1:1 constraint, set will have the same number of elements as the number of matches
        self.assertEqual(result.shape[0], len(set_index_im1))
        self.assertEqual(result.shape[0], len(set_index_im2))

    def test_computation_graph(self):
        """Test that the computation graph is working exactly as the normal
        matching API using 3 images.
        """
        # number of images in test, for which we need to generate descriptors.
        num_images = 3

        # pairs of images to perform matching on
        pairs_list = [(0, 1), (0, 2), (2, 1)]

        descriptor_dimension = random.randint(2, 10)  # descriptor dimension

        # generate descriptors randomly
        descriptors_list = []
        for _ in range(num_images):
            num_descriptors = random.randint(5, 15)

            descriptors_list.append(
                generate_random_binary_descriptors(num_descriptors, descriptor_dimension),
            )

        for (i1, i2) in pairs_list:
            matches_graph = self.matcher.create_computation_graph(
                dask.delayed(descriptors_list[i1]),
                dask.delayed(descriptors_list[i2]),
            )

            with dask.config.set(scheduler="single-threaded"):
                matches = dask.compute(matches_graph)[0]

            # run matching using normal APIs
            expected_matches = self.matcher.match(descriptors_list[i1], descriptors_list[i2])

            np.testing.assert_array_equal(matches, expected_matches)

    def test_pickleable(self):
        """Tests that the matcher object is pickleable (required for dask)."""
        try:
            pickle.dumps(self.matcher)
        except TypeError:
            self.fail("Cannot dump matcher using pickle")

    def __generate_matches_on_random_descriptors(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates a pair of random descriptors and uses the matcher under test to match them.

        Note: using binary descriptors in uint8 format as we want the hamming distances to work

        Returns:
            Matching result on the randomly generated input
            Descriptor input for image #1
            Descriptor input for image #2
        """

        num_descriptors_im1 = random.randint(5, 15)
        num_descriptors_im2 = random.randint(5, 15)

        descriptor_dim = random.randint(2, 10)

        descriptors_im1 = generate_random_binary_descriptors(num_descriptors_im1, descriptor_dim)
        descriptors_im2 = generate_random_binary_descriptors(num_descriptors_im2, descriptor_dim)

        result = self.matcher.match(descriptors_im1, descriptors_im2)

        return result, descriptors_im1, descriptors_im2


def generate_random_binary_descriptors(num_descriptors: int, descriptor_dim: int) -> np.ndarray:
    """Generates random binary descriptors.

    Args:
        num_descriptors (int): number of descriptors to generate
        descriptor_dim (int): length of each descriptor vector

    Returns:
        Generated descriptor
    """
    if num_descriptors == 0:
        return np.array([], dtype=np.uint8)

    return np.random.randint(0, high=2, size=(num_descriptors, descriptor_dim)).astype(np.uint8)


if __name__ == "__main__":
    unittest.main()
