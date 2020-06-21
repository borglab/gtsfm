"""
Tests for frontend's base verifier class.

Authors: Ayush Baid
"""

import random
import unittest
from typing import List, Tuple
from unittest.mock import MagicMock

import dask
import numpy as np

from frontend.verifier.dummy_verifier import DummyVerifier


class TestVerifierBase(unittest.TestCase):
    """
    Unit tests for the Base Verifier class.

    Should be inherited by all verifier unit tests.
    """

    def setUp(self):
        super().setUp()

        self.verifier = DummyVerifier()

    def test_valid_verified_indices(self):
        """Test if valid indices in output."""

        # run matching on a random input pair
        f_matrix, verified_indices, input_features_im1, input_features_im2 = self.__verify_random_inputs()

        if verified_indices.size:
            # check that the indices are not out of bounds
            self.assertTrue(np.all(verified_indices >= 0))
            self.assertTrue(
                np.all(verified_indices < input_features_im1.shape[0]))

    def test_valid_fundamental_matrix(self):
        """Checks the shape of the computed fundamental matrix."""

        for _ in range(10):
            # repeat the experiments 10 times as we might not get the fundamental matrix in every case

            fundamental_matrix, verified_indices, _, _ = self.__verify_random_inputs()

            if verified_indices.size >= self.verifier.min_pts:
                self.assertEqual((3, 3), fundamental_matrix.shape)
            else:
                self.assertIsNone(fundamental_matrix)

    def test_match_empty_input(self):
        """Tests the output when there are no input features."""

        image_shape = [100, 400]

        f_matrix, verified_indices = self.verifier.verify(
            np.array([]), np.array([]), image_shape, image_shape)

        self.assertIsNone(f_matrix)
        self.assertEqual(0, verified_indices.size)

    def test_unique_indices(self):
        """Tests that each index appears atmost once."""

        _, verified_indices, _, _ = self.__verify_random_inputs()

        num_unique_indices = len(set(verified_indices.tolist()))

        self.assertEqual(verified_indices.size, num_unique_indices)

    def test_verify_and_get_features(self):
        """Tests the lookup portion of the verify+lookup API."""

        _, _, matched_features_im1, matched_features_im2 = self.__verify_random_inputs()
        num_input = matched_features_im1.shape[0]

        # mock the verified indices
        mocked_f_matrix = np.random.randn(3, 3)
        mocked_verified_indices = np.random.randint(low=0, high=num_input, size=(
            random.randint(0, num_input)))
        self.verifier.verify = MagicMock(return_value=(
            mocked_f_matrix,
            mocked_verified_indices
        ))

        # run the function
        image_shape = [100, 400]
        f_matrix, verified_features_im1, verified_features_im2 = self.verifier.verify_and_get_features(
            matched_features_im1, matched_features_im2, image_shape, image_shape
        )

        # Confirm by performing manual lookup
        np.testing.assert_array_equal(
            matched_features_im1[mocked_verified_indices], verified_features_im1)
        np.testing.assert_array_equal(
            matched_features_im2[mocked_verified_indices], verified_features_im2)

    def test_computation_graph(self):
        """Tests the computation graph is working exactly as the normal verification API."""

        # Set up 3 pairs of inputs to the verifier
        image_indices = [(0, 1), (4, 3), (2, 5)]
        verifier_input = dict()
        for indices in image_indices:
            verifier_input[indices] = self.__generate_random_input()

        # Convert the input to computation graph
        matched_features_computation_graph = dict()
        image_shapes = [None]*6

        for key, val in verifier_input.items():
            matched_features_computation_graph[key] = dask.delayed(val[:2])
            image_shapes[key[0]] = val[2]
            image_shapes[key[1]] = val[3]

        # generate the computation graph for the verifier
        verifier_output_graph = self.verifier.create_computation_graph(
            matched_features_computation_graph,
            image_shapes
        )

        with dask.config.set(scheduler='single-threaded'):
            computation_graph_results = dask.compute(verifier_output_graph)[0]

            for key, val in verifier_input.items():
                normal_results = self.verifier.verify_and_get_features(
                    val[0], val[1], val[2], val[3]
                )

                dask_results = computation_graph_results[key]

                self.assertEqual(len(normal_results), len(dask_results))

                for idx in range(len(normal_results)):
                    np.testing.assert_array_equal(
                        normal_results[idx], dask_results[idx])

    def __generate_random_features(self, num_features: int, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generates random features within the image bounds.

        Args:
            num_features (int): number of features to generate
            image_shape (Tuple[int, int]): size of the image

        Returns:
            np.ndarray: generated features
        """

        if num_features == 0:
            return np.array([])

        return np.random.randint(
            [0, 0], high=image_shape, size=(num_features, 2)
        ).astype(np.float32)

    def __verify_random_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a pair of image shapes and the features for each image randomly.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                1. computed fundamental matrix
                2. verified indices
                3. input features for image #1
                4. corresponding input features for image #2
        """

        matched_features_im1, matched_features_im2, image_shape_im1, image_shape_im2 = self.__generate_random_input()

        f_matrix, verified_indices = self.verifier.verify(
            matched_features_im1, matched_features_im2, image_shape_im1, image_shape_im2)

        return f_matrix, verified_indices, matched_features_im1, matched_features_im2

    def __generate_random_input(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Generates input for the verify() API randomly.

        Returns:
            Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
                1. Matched features from image #1
                2. Matched features from image #2
                3. Shape of image #1
                4. Shape of image #2
        """

        num_features = random.randint(0, 100)
        image_shape_im1 = [random.randint(100, 400), random.randint(100, 400)]
        image_shape_im2 = [random.randint(100, 400), random.randint(100, 400)]

        matched_features_im1 = self.__generate_random_features(
            num_features, image_shape_im1)
        matched_features_im2 = self.__generate_random_features(
            num_features, image_shape_im2)

        return matched_features_im1, matched_features_im2, image_shape_im1, image_shape_im2
