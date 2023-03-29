"""Tests for frontend's base matcher class.

Authors: Ayush Baid
"""

import pickle
import random
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import dask
import numpy as np

import gtsfm.utils.features as feature_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent / "data"
REAL_FEATURES_PATH = DATA_ROOT_PATH / "set1_lund_door" / "features"


class TestMatcherBase(unittest.TestCase):
    """Unit tests for MatcherBase.

    Should be inherited by all matcher unit tests.
    """

    def setUp(self):
        super().setUp()

        self.matcher: MatcherBase = TwoWayMatcher()

    def __assert_valid_indices(self, match_idxs: np.ndarray, num_keypoints_i1: int, num_keypoints_i2: int) -> None:

        if match_idxs.size:
            self.assertTrue(np.all((match_idxs[:, 0] >= 0) & (match_idxs[:, 0] < num_keypoints_i1)))
            self.assertTrue(np.all((match_idxs[:, 1] >= 0) & (match_idxs[:, 1] < num_keypoints_i2)))

    def __assert_one_to_one_constraint(self, match_idxs: np.ndarray):
        """Asserts that each keypoint is used atmost once."""

        # form sets from the two index columns
        set_index_i1 = set(match_idxs[:, 0].tolist())
        set_index_i2 = set(match_idxs[:, 1].tolist())

        # if we have 1:1 constraint, set will have the same number of elements as the number of matches
        self.assertEqual(len(set_index_i1), match_idxs.shape[0])
        self.assertEqual(len(set_index_i2), match_idxs.shape[0])

    def test_empty_input(self):
        """Tests the matches when there are no descriptors."""

        nonempty_keypoints, _, nonempty_descriptors, _, _, _ = generate_random_input()
        empty_keypoints = Keypoints(coordinates=np.array([]))
        empty_descriptors = np.array([])

        im_shape_i1 = (300, 200)
        im_shape_i2 = (300, 200)

        # no keypoints for just i1
        result = self.matcher.apply(
            empty_keypoints, nonempty_keypoints, empty_descriptors, nonempty_descriptors, im_shape_i1, im_shape_i2
        )
        self.assertEqual(result.size, 0)

        # no keypoints for just i2
        result = self.matcher.apply(
            nonempty_keypoints, empty_keypoints, nonempty_descriptors, empty_descriptors, im_shape_i1, im_shape_i2
        )
        self.assertEqual(result.size, 0)

        # no keypoints for both i1 and i2
        result = self.matcher.apply(
            deepcopy(empty_keypoints),
            deepcopy(empty_keypoints),
            deepcopy(empty_descriptors),
            deepcopy(empty_descriptors),
            im_shape_i1,
            im_shape_i2,
        )
        self.assertEqual(result.size, 0)

    def test_computation_graph(self):
        """Test that the computation graph is working exactly as the normal API"""
        (
            keypoints_i1,
            keypoints_i2,
            descriptors_i1,
            descriptors_i2,
            im_shape_i1,
            im_shape_i2,
        ) = get_features_from_real_images()

        expected_matches = self.matcher.apply(
            keypoints_i1, keypoints_i2, descriptors_i1, descriptors_i2, im_shape_i1, im_shape_i2
        )

        computed_matches_graph = self.matcher.create_computation_graph(
            dask.delayed(keypoints_i1),
            dask.delayed(keypoints_i2),
            dask.delayed(descriptors_i1),
            dask.delayed(descriptors_i2),
            dask.delayed(im_shape_i1),
            dask.delayed(im_shape_i2),
        )
        with dask.config.set(scheduler="single-threaded"):
            computed_matches = dask.compute(computed_matches_graph)[0]

        np.testing.assert_allclose(computed_matches, expected_matches)
        self.__assert_valid_indices(computed_matches, len(keypoints_i1), len(keypoints_i2))
        self.__assert_one_to_one_constraint(computed_matches)

    def test_pickleable(self):
        """Tests that the matcher object is pickleable (required for dask)."""
        try:
            pickle.dumps(self.matcher)
        except TypeError:
            self.fail("Cannot dump matcher using pickle")


def get_features_from_real_images() -> Tuple[
    Keypoints, Keypoints, np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]
]:
    """Load keypoints and descriptors from 2 real images, taken from Olsson's Lund Door dataset."""
    with open(REAL_FEATURES_PATH / "keypoints_0.pkl", "rb") as f:
        keypoints_i1 = pickle.load(f)
    with open(REAL_FEATURES_PATH / "keypoints_1.pkl", "rb") as f:
        keypoints_i2 = pickle.load(f)
    descriptors_i1 = np.load(REAL_FEATURES_PATH / "descriptors_0.npy")
    descriptors_i2 = np.load(REAL_FEATURES_PATH / "descriptors_1.npy")

    # shape as (height, width)
    im_shape_i1 = (1936, 1296)
    im_shape_i2 = (1936, 1296)
    return keypoints_i1, keypoints_i2, descriptors_i1, descriptors_i2, im_shape_i1, im_shape_i2


def generate_random_input() -> Tuple[Keypoints, Keypoints, np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Generates random keypoints and descriptors for a pair of images."""

    num_keypoints_i1 = random.randint(5, 15)
    num_keypoints_i2 = random.randint(5, 15)

    descriptor_dim = random.randint(2, 10)

    im_shape_i1 = (300, 100)
    im_shape_i2 = (150, 200)
    keypoints_i1 = feature_utils.generate_random_keypoints(num_keypoints_i1, im_shape_i1)
    keypoints_i2 = feature_utils.generate_random_keypoints(num_keypoints_i2, im_shape_i2)
    descriptors_i1 = generate_random_binary_descriptors(num_keypoints_i1, descriptor_dim)
    descriptors_i2 = generate_random_binary_descriptors(num_keypoints_i2, descriptor_dim)

    return keypoints_i1, keypoints_i2, descriptors_i1, descriptors_i2, im_shape_i1, im_shape_i2


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
