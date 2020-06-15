"""
Dummy matcher which produces random results.

Authors: Ayush Baid
"""
import numpy as np

from frontend.matcher.matcher_base import MatcherBase


class DummyMatcher(MatcherBase):
    """
    Dummy matcher to be used for tests.
    """

    def match(self, descriptors_im1: np.ndarray, descriptors_im2: np.ndarray) -> np.ndarray:
        """
        Match a pair of descriptors.

        Refer to the doc in the parent class for output format.

        Args:
            descriptors_im1 (np.ndarray): descriptors from image #1
            descriptors_im2 (np.ndarray): descriptors from image #2

        Returns:
            np.ndarray: match indices (sorted by confidence)
        """

        # check if we have non-zero descriptors in the both image
        if descriptors_im1.size == 0 or descriptors_im2.size == 0:
            return np.array([], dtype=np.int)

        # set a random seed using descriptor data for repeatibility
        np.random.seed(
            int(1000*(np.sum(descriptors_im1, axis=None) +
                      np.sum(descriptors_im2, axis=None)) % (2 ^ 32))
        )

        # get the number of entries in the input
        num_input_1 = descriptors_im1.shape[0]
        num_input_2 = descriptors_im2.shape[0]

        # get the number of matches we will output
        num_matches = np.random.randint(
            low=0, high=min(num_input_1, num_input_2))

        # randomly sample index for the matches
        result = np.empty((num_matches, 2))
        result[:, 0] = np.random.choice(
            num_input_1, size=num_matches, replace=False)
        result[:, 1] = np.random.choice(
            num_input_2, size=num_matches, replace=False)

        return result.astype(np.uint32)
