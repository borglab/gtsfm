"""
Degensac.

Authors: Ayush Baid
"""

from typing import Optional, Tuple

import numpy as np
import pydegensac
from gtsam import Cal3Bundler, EssentialMatrix

from frontend.verifier.verifier_base import VerifierBase
import utils.verification as verification_utils


class Degensac(VerifierBase):
    def __init__(self):
        super().__init__(min_pts=8)

    def verify_with_exact_intrinsics(
        self,
        features_im1: np.ndarray,
        features_im2: np.ndarray,
        match_indices: np.ndarray,
        camera_instrinsics_im1: Cal3Bundler,
        camera_instrinsics_im2: Cal3Bundler,
    ) -> Tuple[Optional[EssentialMatrix], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is prefered when camera intrinsics are known. The
        feature coordinates are normalized and the essential matrix is directly
        estimated.

        Args:
            features_im1: detected features in image #1, of shape (N1, 2+).
            features_im2: detected features in image #2, of shape (N2, 2+).
            match_indices: matches as indices of features from both images, of
                           shape (N3, 2), where N3 <= min(N1, N2).
            camera_instrinsics_im1: intrinsics for image #1.
            camera_instrinsics_im2: intrinsics for image #2.

        Returns:
            Estimated essential matrix im2_E_im1, or None if it cannot be 
                estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3.
                These indices are subset of match_indices.
        """
        raise NotImplementedError(
            'Degensac verifier cannot compute essential matrix directly using' +
            ' exact intrinsics')

    def verify_with_approximate_intrinsics(
        self,
        features_im1: np.ndarray,
        features_im2: np.ndarray,
        match_indices: np.ndarray,
        camera_instrinsics_im1: Cal3Bundler,
        camera_instrinsics_im2: Cal3Bundler,
    ) -> Tuple[Optional[EssentialMatrix], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is prefered when camera intrinsics are approximate
        (i.e from image size/exif). The feature coordinates are used to compute
        the fundamental matrix, which is then converted to the essential matrix.

        Args:
            features_im1: detected features in image #1, of shape (N1, 2+).
            features_im2: detected features in image #2, of shape (N2, 2+).
            match_indices: matches as indices of features from both images, of
                           shape (N3, 2), where N3 <= min(N1, N2).
            camera_instrinsics_im1: intrinsics for image #1.
            camera_instrinsics_im2: intrinsics for image #2.

        Returns:
            Estimated essential matrix im2_E_im1, or None if it cannot be
                estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3.
                These indices are subset of match_indices.
        """
        im2_F_im1, mask = pydegensac.findFundamentalMatrix(
            features_im1[match_indices[:, 0], :2],
            features_im2[match_indices[:, 1], :2],
        )

        inlier_idx = np.where(mask.ravel() == 1)[0]

        e_matrix = verification_utils.fundamental_matrix_to_essential_matrix(
            im2_F_im1,
            camera_instrinsics_im1,
            camera_instrinsics_im2
        )

        im2_E_im1 = verification_utils.cast_essential_matrix_to_gtsam(
            e_matrix,
            features_im1[match_indices[inlier_idx, 0], :2],
            features_im2[match_indices[inlier_idx, 1], :2],
        )

        return im2_E_im1, inlier_idx
