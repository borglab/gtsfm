"""
Joint Matcher-Verifier combination using stand-alone matching and verification modules.

Authors: Ayush Baid
"""

from typing import Tuple

import numpy as np

from frontend.matcher.matcher_base import MatcherBase
from frontend.matcher_verifier.matcher_verifier_base import MatcherVerifierBase
from frontend.verifier.verifier_base import VerifierBase


class CombinationMatcherVerifier(MatcherVerifierBase):
    """A wrapper to combine stand-alone matcher and verifier."""

    def __init__(self, matcher: MatcherBase, verifier: VerifierBase):
        """
        Initialize from stand-alone objects

        Args:
            matcher (MatcherBase): the matcher to combine
            verifier (VerifierBase): the verifier to combine
        """
        self.matcher = matcher
        self.verifier = verifier

    def match_and_verify(self,
                         features_im1: np.ndarray,
                         features_im2: np.ndarray,
                         descriptors_im1: np.ndarray,
                         descriptors_im2: np.ndarray,
                         image_shape_im1: Tuple[int, int],
                         image_shape_im2: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Matches the features (using their corresponding descriptors) to return geometrically verified outlier-free correspondences as indices of input features.

        Args:
            features_im1 (np.ndarray): features from image #1
            features_im2 (np.ndarray): features from image #2
            descriptors_im1 (np.ndarray): corresponding descriptors from image #1
            descriptors_im2 (np.ndarray): corresponding descriptors from image #1
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2

        Returns:
            np.ndarray: estimated fundamental matrix
            np.ndarray: verified correspondences as index of the input features in a Nx2 array
        """

        matched_features_im1, matched_features_im2 = self.matcher.match_and_get_features(
            features_im1, features_im2, image_shape_im1, image_shape_im2)

        return self.verifier.verify(matched_features_im1, matched_features_im2, image_shape_im1, image_shape_im2)

    def match_and_verify_and_get_features(self,
                                          features_im1: np.ndarray,
                                          features_im2: np.ndarray,
                                          descriptors_im1: np.ndarray,
                                          descriptors_im2: np.ndarray,
                                          image_shape_im1: Tuple[int, int],
                                          image_shape_im2: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs the match_and_verify functionality and fetches the actual features corresponding to indices of verified correspondenses.

        Args:
            features_im1 (np.ndarray): features from image #1
            features_im2 (np.ndarray): features from image #2
            descriptors_im1 (np.ndarray): corresponding descriptors from image #1
            descriptors_im2 (np.ndarray): corresponding descriptors from image #1
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2

        Returns:
            np.ndarray: estimated fundamental matrix
            np.ndarray: the features from image #1 which serve as verified correspondences.
            np.ndarray: the corresponding features from image #2 which serve as verified correspondences.
        """

        matched_features_im1, matched_features_im2 = self.matcher.match_and_get_features(
            features_im1, features_im2, descriptors_im1, descriptors_im2)

        return self.verifier.verify_and_get_features(matched_features_im1, matched_features_im2, image_shape_im1, image_shape_im2)
