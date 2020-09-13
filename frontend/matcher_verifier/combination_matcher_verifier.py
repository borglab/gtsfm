"""Joint Matcher-Verifier combination using stand-alone matching and
verification modules.

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
                         image_shape_im2: Tuple[int, int],
                         camera_instrinsics_im1: np.ndarray = None,
                         camera_instrinsics_im2: np.ndarray = None,
                         distance_type: str = 'euclidean'
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Matches the features (using their corresponding descriptors) to
        return geometrically verified outlier-free correspondences as indices of
        input features.

        Note:
        1. The function computes the fundamental matrix if intrinsics are not
           provided. Otherwise, it computes the essential matrix.

        Args:
            features_im1 (np.ndarray): features from image #1
            features_im2 (np.ndarray): features from image #2
            descriptors_im1 (np.ndarray): corr. descriptors from image #1
            descriptors_im2 (np.ndarray): corr. descriptors from image #2
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2
            camera_instrinsics_im1 (np.ndarray, optional): Camera intrinsics
                matrix for image #1. Defaults to None.
            camera_instrinsics_im2 (np.ndarray, optional): Camera intrinsics
                matris for image #2. Default to None
            distance_type (str, optional): the space to compute the distance
                                           between descriptors. Defaults to
                                           'euclidean'.

        Returns:
            np.ndarray: estimated fundamental/essential matrix
            np.ndarray: index of the input features which are verified (Nx2)
        """

        matched_features_im1, matched_features_im2 = \
            self.matcher.match_and_get_features(
                features_im1, features_im2,
                descriptors_im1, descriptors_im2,
                distance_type)

        return self.verifier.verify(matched_features_im1,
                                    matched_features_im2,
                                    image_shape_im1,
                                    image_shape_im2,
                                    camera_instrinsics_im1,
                                    camera_instrinsics_im2)

    def match_and_verify_and_get_features(
        self,
        features_im1: np.ndarray,
        features_im2: np.ndarray,
        descriptors_im1: np.ndarray,
        descriptors_im2: np.ndarray,
        image_shape_im1: Tuple[int, int],
        image_shape_im2: Tuple[int, int],
        camera_instrinsics_im1: np.ndarray = None,
        camera_instrinsics_im2: np.ndarray = None,
        distance_type: str = 'euclidean'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calls the match_and_verify function to return actual features
        instead of indices.

        Args:
            features_im1 (np.ndarray): features from image #1
            features_im2 (np.ndarray): features from image #2
            descriptors_im1 (np.ndarray): corr. descriptors from image #1
            descriptors_im2 (np.ndarray): corr. descriptors from image #2
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2
            camera_instrinsics_im1 (np.ndarray, optional): Camera intrinsics
                matrix for image #1. Defaults to None.
            camera_instrinsics_im2 (np.ndarray, optional): Camera intrinsics
                matris for image #2. Default to None
            distance_type (str, optional): the space to compute the distance
                                           between descriptors. Defaults to
                                           'euclidean'.

        Returns:
            np.ndarray: estimated fundamental/essential matrix
            np.ndarray: verified features from image #1
            np.ndarray: corresponding verified features from image #2
        """

        matched_features_im1, matched_features_im2 = \
            self.matcher.match_and_get_features(
                features_im1, features_im2,
                descriptors_im1, descriptors_im2,
                distance_type)

        return self.verifier.verify_and_get_features(matched_features_im1,
                                                     matched_features_im2,
                                                     image_shape_im1,
                                                     image_shape_im2,
                                                     camera_instrinsics_im1,
                                                     camera_instrinsics_im2)
