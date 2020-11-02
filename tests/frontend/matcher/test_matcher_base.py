"""
Tests for frontend's base matcher class.

Authors: Ayush Baid
"""

import pickle
import random
import unittest
from typing import Tuple

import dask
import numpy as np

from frontend.matcher.dummy_matcher import DummyMatcher


class TestMatcherBase(unittest.TestCase):
    """
    Unit tests for the Base Matcher class.

    Should be inherited by all matcher unit tests.
    """

    def setUp(self):
        super().setUp()

        self.matcher = DummyMatcher()

    def test_match_valid_indices(self):
        """
        Tests if valid indices in output.
        """

        # run matching on a random input pair
        result, descriptors_im1, descriptors_im2 = self.__generate_matches_on_random_descriptors()

        if result.size:
            # check that the match indices are out of bounds
            self.assertTrue(np.all((result[:, 0] >= 0) & (
                result[:, 0] < descriptors_im1.shape[0])))
            self.assertTrue(np.all((result[:, 1] >= 0) & (
                result[:, 1] < descriptors_im2.shape[0])))

    def test_empty_input(self):
        """
        Tests the matches when there are no descriptors.
        """

        num_descriptors = random.randint(5, 15)

        descriptor_length = random.randint(2, 10)

        descriptors = self.__generate_random_binary_descriptors(
            num_descriptors, descriptor_length)

        # the first descriptor is empty
        result = self.matcher.match(self.__generate_random_binary_descriptors(0, descriptor_length),
                                    descriptors)

        self.assertEqual(0, result.size)

        # the second descriptor is empty
        result = self.matcher.match(descriptors,
                                    self.__generate_random_binary_descriptors(0, descriptor_length))

        self.assertEqual(0, result.size)

        # both descriptors are empty
        result = self.matcher.match(self.__generate_random_binary_descriptors(0, descriptor_length),
                                    self.__generate_random_binary_descriptors(0, descriptor_length))

        self.assertEqual(0, result.size)

    def test_one_to_one_constraint(self):
        """
        Tests that each index from an image is used atmost once.
        """

        # get a result
        result, _, _ = self.__generate_matches_on_random_descriptors()

        # form sets from the two index columns
        set_index_im1 = set(result[:, 0].tolist())
        set_index_im2 = set(result[:, 1].tolist())

        # if we have 1:1 constraint, set will have the same number of elements as the number of matches
        self.assertEqual(result.shape[0], len(set_index_im1))
        self.assertEqual(result.shape[0], len(set_index_im2))

    def test_match_and_get_features(self):
        """
        Testing the match+lookup API to verify lookup.
        """

        # run matching on a random input pair
        match_indices, descriptors_im1, descriptors_im2 = self.__generate_matches_on_random_descriptors()

        # generate random features for input
        features_im1 = np.random.randn(descriptors_im1.shape[0], 3)
        features_im2 = np.random.randn(descriptors_im2.shape[0], 3)

        # fetch matched features directly from the matcher
        matched_features_im1, matched_features_im2 = self.matcher.match_and_get_features(
            features_im1, features_im2,
            descriptors_im1, descriptors_im2)

        # Confirm by performing manual lookup
        np.testing.assert_array_equal(
            features_im1[match_indices[:, 0]], matched_features_im1)
        np.testing.assert_array_equal(
            features_im2[match_indices[:, 1]], matched_features_im2)

    def test_computation_graph(self):
        """
        Test that the computation graph is working exactly as the normal matching API using 3 images.
        """

        # generate three random descriptors and their features
        num_descriptors_im1 = random.randint(5, 15)
        num_descriptors_im2 = random.randint(5, 15)
        num_descriptors_im3 = random.randint(5, 15)

        descriptor_length = random.randint(2, 10)

        descriptor_list = [
            self.__generate_random_binary_descriptors(
                num_descriptors_im1, descriptor_length),
            self.__generate_random_binary_descriptors(
                num_descriptors_im2, descriptor_length),
            self.__generate_random_binary_descriptors(
                num_descriptors_im3, descriptor_length)
        ]

        features_list = [
            np.random.rand(num_descriptors_im1, 3),
            np.random.rand(num_descriptors_im2, 3),
            np.random.rand(num_descriptors_im3, 3),
        ]

        # create computation graphs
        detection_description_graph = [
            dask.delayed((x, y)) for x, y in zip(features_list, descriptor_list)
        ]

        matcher_graph = self.matcher.create_computation_graph(
            [(0, 1), (0, 2), (2, 1)],
            detection_description_graph)

        # run it in sequential mode
        results = []
        with dask.config.set(scheduler='single-threaded'):
            results = dask.compute(matcher_graph)[0]

        # check the number of pairs in the results
        self.assertEqual(len(descriptor_list), len(results))

        # check every pair of results

        for image_indices in matcher_graph.keys():
            dask_matches = results[image_indices]
            normal_matches = self.matcher.match(
                descriptor_list[image_indices[0]],
                descriptor_list[image_indices[1]]
            )

            np.testing.assert_array_equal(
                normal_matches, dask_matches
            )

    def test_pickleable(self):
        """
        Tests that the matcher object is pickleable (required for dask)
        """
        try:
            pickle.dumps(self.matcher)
        except TypeError:
            self.fail("Cannot dump matcher using pickle")

    def __generate_random_binary_descriptors(self, num_descriptors: int, descriptor_length: int) -> np.ndarray:
        """
        Generates random binary descriptors.

        Args:
            num_descriptors (int): number of descriptors to generate
            descriptor_length (int): length of each descriptor vector

        Returns:
            np.ndarray: generated descriptor
        """
        if num_descriptors == 0:
            return np.array([], dtype=np.uint8)

        return np.random.randint(
            0, high=2, size=(num_descriptors, descriptor_length)
        ).astype(np.uint8)

    def __generate_matches_on_random_descriptors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a pair of random descriptors and the matching result on them.

        Note: using binary descriptors in uint8 format as we want the hamming distances to work

        Returns:
            Tuple[np.ndarray,np.ndarray,np.ndarray]: 1. matching result on the randomly generated input
                                                     2. descriptor input for image #1
                                                     3. descriptor input for image #2
        """

        num_descriptors_im1 = random.randint(5, 15)
        num_descriptors_im2 = random.randint(5, 15)

        descriptor_length = random.randint(2, 10)

        descriptors_im1 = self.__generate_random_binary_descriptors(
            num_descriptors_im1, descriptor_length)
        descriptors_im2 = self.__generate_random_binary_descriptors(
            num_descriptors_im2, descriptor_length)

        result = self.matcher.match(descriptors_im1, descriptors_im2)

        return result, descriptors_im1, descriptors_im2
