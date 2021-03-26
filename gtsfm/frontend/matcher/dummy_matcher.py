"""Dummy matcher which produces random results.

Authors: Ayush Baid
"""
import numpy as np

from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.matcher.matcher_base import MatchingDistanceType


class DummyMatcher(MatcherBase):
    """Dummy matcher to be used for tests."""

    def match(
        self,
        descriptors_im1: np.ndarray,
        descriptors_im2: np.ndarray,
        distance_type: MatchingDistanceType = MatchingDistanceType.EUCLIDEAN,
    ) -> np.ndarray:
        """Match descriptor vectors randomly.

        Output format:
        1. Each row represents a match.
        2. First column represents descriptor index from image #1.
        3. Second column represents descriptor index from image #2.
        4. Matches are sorted in descending order of the confidence (score).

        Args:
            descriptors_im1: descriptors from image #1, of shape (N1, D).
            descriptors_im2: descriptors from image #2, of shape (N2, D).
            distance_type (optional): the space to compute the distance between descriptors. Defaults to
                                      MatchingDistanceType.EUCLIDEAN.

        Returns:
            Match indices (sorted by confidence), as matrix of shape (N, 2), where N < min(N1, N2).
        """

        # check if we have non-zero descriptors in the both image
        if descriptors_im1.size == 0 or descriptors_im2.size == 0:
            return np.array([], dtype=np.uint32)

        # set a random seed using descriptor data for repeatibility
        np.random.seed(int(1000 * (np.sum(descriptors_im1, axis=None) + np.sum(descriptors_im2, axis=None)) % (2 ^ 32)))

        # get the number of entries in the input
        num_input_1 = descriptors_im1.shape[0]
        num_input_2 = descriptors_im2.shape[0]

        # get the number of matches we will output
        num_matches = np.random.randint(low=0, high=min(num_input_1, num_input_2))

        # randomly sample index for the matches
        result = np.empty((num_matches, 2))
        result[:, 0] = np.random.choice(num_input_1, size=num_matches, replace=False)
        result[:, 1] = np.random.choice(num_input_2, size=num_matches, replace=False)

        return result.astype(np.uint32)
