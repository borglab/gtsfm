"""
Dummy matcher which produces random results.

Authors: Ayush Baid
"""
from typing import Tuple

import numpy as np

from frontend.verifier.base_verifier import VerifierBase


class DummyVerifier(VerifierBase):
    """A dummy verifier which produces random results"""

    def __init__(self):
        super().__init__(min_pts=8)

    def verify(self,
               matched_features_im1: np.ndarray,
               matched_features_im2: np.ndarray,
               image_shape_im1: Tuple[int, int],
               image_shape_im2: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the geometric verification of the matched features.

        Note:
        1. The number of input features from image #1 are the same as the number from image #2

        Args:
            matched_features_im1 (np.ndarray): matched features from image #1
            matched_features_im2 (np.ndarray): matched features from image #2
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2

        Returns:
            np.ndarray: estimated fundamental matrix
            np.ndarray: index of the match features which are verified
        """
        fundamental_matrix = None
        verified_indices = np.array([], dtype=np.uint32)

        # check if we dont have the minimum number of points
        if matched_features_im1.size <= self.min_pts:
            return fundamental_matrix, verified_indices

        # set a random seed using descriptor data for repeatibility
        np.random.seed(
            int(1000*(matched_features_im1[0, 0] +
                      matched_features_im2[0, 0]) % (2 ^ 32))
        )

        # get the number of entries in the input
        num_matches = matched_features_im1.shape[0]

        # get the number of verified_pts we will output
        num_verifier_pts = np.random.randint(
            low=0, high=num_matches)

        # randomly sample the indices for matches which will be verified
        verified_indices = np.random.choice(
            num_matches, num_verifier_pts, replace=False).astype(np.uint32)

        if num_verifier_pts >= self.min_pts:
            fundamental_matrix = np.random.randn(3, 3)

        return fundamental_matrix, verified_indices
