"""
Degensac.

Authors: Ayush Baid
"""

from typing import Tuple

import numpy as np
import pydegensac

from frontend.verifier.verifier_base import VerifierBase


class Degensac(VerifierBase):
    def __init__(self):
        super().__init__(min_pts=8)

    def verify(self,
               matched_features_im1: np.ndarray,
               matched_features_im2: np.ndarray,
               image_shape_im1: Tuple[int, int],
               image_shape_im2: Tuple[int, int],
               camera_instrinsics_im1: np.ndarray = None,
               camera_instrinsics_im2: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Perform the geometric verification of the matched features.

        Note:
        1. The number of input features from image #1 and image #2 are equal.
        2. The function computes the fundamental matrix if intrinsics are not
            provided. Otherwise, it computes the essential matrix.

        Args:
            matched_features_im1 (np.ndarray): matched features from image #1
            matched_features_im2 (np.ndarray): matched features from image #2
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2
            camera_instrinsics_im1 (np.ndarray, optional): Camera intrinsics
                matrix for image #1. Defaults to None.
            camera_instrinsics_im2 (np.ndarray, optional): Camera intrinsics
                matris for image #2. Default to None

        Returns:
            np.ndarray: estimated fundamental/essential matrix
            np.ndarray: index of the match features which are verified
        """

        if camera_instrinsics_im1 is not None and camera_instrinsics_im2 is not None:
            raise NotImplementedError(
                "Degensac for essential matrix is not implemented")

        F, mask = pydegensac.findFundamentalMatrix(
            matched_features_im1[:, :2],
            matched_features_im2[:, :2],
        )

        inlier_idx = np.where(mask.ravel() == 1)[0]

        return F, inlier_idx
