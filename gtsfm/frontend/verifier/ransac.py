"""
RANSAC verifier implementation.

The verifier is the 5-Pt/8-pt Algorithm with RANSAC and is implemented by wrapping over 3rd party implementation.\

References: 
- David Nistér. An efficient solution to the five-point relative pose problem. TPAMI, 2004.
- https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga13f7e34de8fa516a686a56af1196247f
- https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gae420abc34eaa03d0c6a67359609d8429

Authors: John Lambert
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from gtsam import Cal3Bundler, Rot3, Unit3

import gtsfm.utils.features as feature_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.verifier.verifier_base import VerifierBase


NORMALIZED_COORD_RANSAC_THRESH = 0.001  # TODO: hyperparameter to tune
PIXEL_COORD_RANSAC_THRESH = 0.5  # TODO: hyperparameter to tune
DEFAULT_RANSAC_SUCCESS_PROB = 0.9999

logger = logger_utils.get_logger()


class Ransac(VerifierBase):
    def verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Performs verification of correspondences between two images to recover the relative pose and indices of
        verified correspondences.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.
        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3. These are subset of match_indices.
        """
        if match_indices.shape[0] < self._min_matches:
            return self._failure_result

        if self._use_intrinsics_in_verification:
            uv_norm_i1 = feature_utils.normalize_coordinates(keypoints_i1.coordinates, camera_intrinsics_i1)
            uv_norm_i2 = feature_utils.normalize_coordinates(keypoints_i2.coordinates, camera_intrinsics_i2)
            K = np.eye(3)

            i2Ei1, inlier_mask = cv2.findEssentialMat(
                uv_norm_i1[match_indices[:, 0]],
                uv_norm_i2[match_indices[:, 1]],
                K,
                method=cv2.RANSAC,
                threshold=NORMALIZED_COORD_RANSAC_THRESH,
                prob=DEFAULT_RANSAC_SUCCESS_PROB,
            )
        else:
            i2Fi1, inlier_mask = cv2.findFundamentalMat(
                keypoints_i1.extract_indices(match_indices[:, 0]).coordinates,
                keypoints_i2.extract_indices(match_indices[:, 1]).coordinates,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=PIXEL_COORD_RANSAC_THRESH,
                confidence=DEFAULT_RANSAC_SUCCESS_PROB,
                maxIters=10000,
            )

            i2Ei1 = verification_utils.fundamental_to_essential_matrix(
                i2Fi1, camera_intrinsics_i1, camera_intrinsics_i2
            )

        inlier_idxs = np.where(inlier_mask.ravel() == 1)[0]

        (i2Ri1, i2Ui1) = verification_utils.recover_relative_pose_from_essential_matrix(
            i2Ei1,
            keypoints_i1.coordinates[match_indices[inlier_idxs, 0]],
            keypoints_i2.coordinates[match_indices[inlier_idxs, 1]],
            camera_intrinsics_i1,
            camera_intrinsics_i2,
        )

        return i2Ri1, i2Ui1, match_indices[inlier_idxs]
