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

import cv2
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Rot3, Unit3

import utils.features as feature_utils
import utils.verification as verification_utils
from frontend.verifier.verifier_base import VerifierBase
from common.keypoints import Keypoints

# minimum matches required for computing the E-matrix
NUM_MATCHES_REQ_E_MATRIX = 5


class Ransac(VerifierBase):
    def __init__(self):
        super().__init__(min_pts=NUM_MATCHES_REQ_E_MATRIX)

    def verify_with_exact_intrinsics(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is preferred when camera intrinsics are known. The
        feature coordinates are normalized and the essential matrix is directly
        estimated.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
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
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is preferred when camera intrinsics are approximate
        (i.e from image size/exif). The feature coordinates are used to compute
        the fundamental matrix, which is then converted to the essential matrix.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
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

        # check if we don't have the minimum number of points
        if match_indices.shape[0] < self.min_pts:
            return i2Ei1, verified_indices

        uv_norm_i1 = feature_utils.normalize_coordinates(
            keypoints_i1.coordinates,
            camera_intrinsics_i1
        )
        uv_norm_i2 = feature_utils.normalize_coordinates(
            keypoints_i2.coordinates,
            camera_intrinsics_i2
        )
        K = np.eye(3)

        i2Ei1, inlier_mask = cv2.findEssentialMat(
            uv_norm_i1[match_indices[:, 0]],
            uv_norm_i2[match_indices[:, 1]],
            K,
            method=cv2.RANSAC,
            threshold=0.1
        )
        inlier_idxs = np.where(inlier_mask.ravel() == 1)[0]

        i2Ri1, i2Ui1 = \
            verification_utils.recover_relative_pose_from_essential_matrix(
                i2Ei1,
                keypoints_i1.coordinates[match_indices[inlier_idxs, 0]],
                keypoints_i2.coordinates[match_indices[inlier_idxs, 1]],
                camera_intrinsics_i1,
                camera_intrinsics_i2
            )

        return i2Ri1, i2Ui1, match_indices[inlier_idxs]
