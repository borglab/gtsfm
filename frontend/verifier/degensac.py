"""
Locally Optimized (LO) Degensac verifier implementation.

The verifier is a combination of 'Locally Optimized Ransac' and 'Two-view
Geometry Estimation Unaffected by a Dominant Plane' and is implemented by
wrapping over 3rd party implementation.

References:
- https://link.springer.com/chapter/10.1007/978-3-540-45243-0_31
- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.466.2719&rep=rep1&type=pdf
- https://github.com/ducha-aiki/pyransac

Authors: Ayush Baid
"""

from typing import Optional, Tuple

import numpy as np
import pydegensac
from gtsam import Cal3Bundler, EssentialMatrix

from frontend.verifier.verifier_base import VerifierBase
import utils.verification as verification_utils

# minimum matches required for computing the F-matrix
NUM_MATCHES_REQ_F_MATRIX = 8


class Degensac(VerifierBase):
    def __init__(self):
        super().__init__(min_pts=NUM_MATCHES_REQ_F_MATRIX)

    def verify_with_exact_intrinsics(
        self,
        keypoints_i1: np.ndarray,
        keypoints_i2: np.ndarray,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[EssentialMatrix], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is prefered when camera intrinsics are known. The
        feature coordinates are normalized and the essential matrix is directly
        estimated.

        Args:
            keypoints_i1: detected features in image #i1, of shape (N1, 2+).
            keypoints_i2: detected features in image #i2, of shape (N2, 2+).
            match_indices: matches as indices of features from both images, of
                           shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            Estimated essential matrix i2Ei1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3.
                These indices are subset of match_indices.
        """
        print('WARNING: Degensac verifier cannot compute essential matrix directly using exact intrinsics')

        return self.verify_with_approximate_intrinsics(
            keypoints_i1,
            keypoints_i2,
            match_indices,
            camera_intrinsics_i1,
            camera_intrinsics_i2
        )

    def verify_with_approximate_intrinsics(
        self,
        keypoints_i1: np.ndarray,
        keypoints_i2: np.ndarray,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[EssentialMatrix], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is prefered when camera intrinsics are approximate
        (i.e from image size/exif). The feature coordinates are used to compute
        the fundamental matrix, which is then converted to the essential matrix.

        Args:
            keypoints_i1: detected features in image #i1, of shape (N1, 2+).
            keypoints_i2: detected features in image #i2, of shape (N2, 2+).
            match_indices: matches as indices of features from both images, of
                           shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            Estimated essential matrix i2Ei1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3.
                These indices are subset of match_indices.
        """
        i2Ei1 = None
        verified_indices = np.array([], dtype=np.uint32)

        # check if we dont have the minimum number of points
        if match_indices.shape[0] <= self.min_pts:
            return i2Ei1, verified_indices

        i2Fi1, mask = pydegensac.findFundamentalMatrix(
            keypoints_i1[match_indices[:, 0], :2],
            keypoints_i2[match_indices[:, 1], :2],
        )

        inlier_idx = np.where(mask.ravel() == 1)[0]

        e_matrix = verification_utils.fundamental_matrix_to_essential_matrix(
            i2Fi1,
            camera_intrinsics_i1,
            camera_intrinsics_i2
        )

        i2Ei1 = verification_utils.cast_essential_matrix_to_gtsam(
            e_matrix,
            keypoints_i1[match_indices[inlier_idx, 0], :2],
            keypoints_i2[match_indices[inlier_idx, 1], :2],
            camera_intrinsics_i1,
            camera_intrinsics_i2
        )

        return i2Ei1, match_indices[inlier_idx]
