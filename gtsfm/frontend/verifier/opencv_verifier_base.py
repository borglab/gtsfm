"""Base-class for OpenCV verifier implementations.

Authors: John Lambert
"""

import abc
from typing import Optional, Tuple

import numpy as np
from gtsam import Cal3Bundler, Rot3, Unit3

import gtsfm.utils.features as feature_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.verifier.verifier_base import NUM_MATCHES_REQ_E_MATRIX, NUM_MATCHES_REQ_F_MATRIX, VerifierBase

logger = logger_utils.get_logger()


class OpencvVerifierBase(VerifierBase):
    def __init__(
        self,
        use_intrinsics_in_verification: bool,
        estimation_threshold_px: float,
    ) -> None:
        """Initializes the verifier.

        Args:
            use_intrinsics_in_verification: Flag to perform keypoint normalization and compute the essential matrix
                instead of fundamental matrix. This should be preferred when the exact intrinsics are known as opposed
                to approximating them from exif data.
            estimation_threshold_px: maximum distance (in pixels) to consider a match an inlier, under squared
                Sampson distance.
        """
        self._use_intrinsics_in_verification = use_intrinsics_in_verification
        self._estimation_threshold_px = estimation_threshold_px
        self._min_matches = (
            NUM_MATCHES_REQ_E_MATRIX if self._use_intrinsics_in_verification else NUM_MATCHES_REQ_F_MATRIX
        )

        # for failure, i2Ri1 = None, and i2Ui1 = None, and no verified correspondences, and inlier_ratio_est_model = 0
        self._failure_result = (None, None, np.array([], dtype=np.uint64), 0.0)

    def verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, float]:
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
            Inlier ratio w.r.t. the estimated model, i.e. #ransac inliers / # putative matches.
        """
        if match_indices.shape[0] < self._min_matches:
            return self._failure_result

        if self._use_intrinsics_in_verification:
            uv_norm_i1 = feature_utils.normalize_coordinates(keypoints_i1.coordinates, camera_intrinsics_i1)
            uv_norm_i2 = feature_utils.normalize_coordinates(keypoints_i2.coordinates, camera_intrinsics_i2)

            # OpenCV can fail here, for some reason
            if match_indices.shape[0] < 6:
                return self._failure_result

            if np.amax(match_indices[:, 1]) >= uv_norm_i2.shape[0]:
                print("Out of bounds access w/ keypoints", keypoints_i2.coordinates[:10])
            if np.amax(match_indices[:, 0]) >= uv_norm_i1.shape[0]:
                print("Out of bounds access w/ keypoints", keypoints_i1.coordinates[:10])

            # Use larger focal length, among the two choices, to yield a stricter threshold as (threshold_px / fx).
            fx = max(camera_intrinsics_i1.K()[0, 0], camera_intrinsics_i2.K()[0, 0])
            i2Ei1, inlier_mask = self.estimate_E(
                uv_norm_i1=uv_norm_i1, uv_norm_i2=uv_norm_i2, match_indices=match_indices, fx=fx
            )
        else:
            i2Fi1, inlier_mask = self.estimate_F(
                keypoints_i1=keypoints_i1, keypoints_i2=keypoints_i2, match_indices=match_indices
            )
            i2Ei1 = verification_utils.fundamental_to_essential_matrix(
                i2Fi1, camera_intrinsics_i1, camera_intrinsics_i2
            )

        inlier_idxs = np.where(inlier_mask.ravel() == 1)[0]

        v_corr_idxs = match_indices[inlier_idxs]
        inlier_ratio_est_model = np.mean(inlier_mask)
        (i2Ri1, i2Ui1) = verification_utils.recover_relative_pose_from_essential_matrix(
            i2Ei1,
            keypoints_i1.coordinates[v_corr_idxs[:, 0]],
            keypoints_i2.coordinates[v_corr_idxs[:, 1]],
            camera_intrinsics_i1,
            camera_intrinsics_i2,
        )
        return i2Ri1, i2Ui1, v_corr_idxs, inlier_ratio_est_model

    @abc.abstractmethod
    def estimate_E(
        self, uv_norm_i1: np.ndarray, uv_norm_i2: np.ndarray, match_indices: np.ndarray, fx: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the Essential matrix from correspondences.

        Args:
            uv_norm_i1: normalized coordinates of detected features in image #i1.
            uv_norm_i2: normalized coordinates of detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2),
               given N1 features from image 1, and N2 features from image 2.
            fx: focal length (in pixels) in horizontal direction.

        Returns:
            i2Ei1: Essential matrix, as 3x3 array.
            inlier_mask: boolean array of shape (N3,) indicating inlier matches.
        """

    @abc.abstractmethod
    def estimate_F(
        self, keypoints_i1: Keypoints, keypoints_i2: Keypoints, match_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the Fundamental matrix from correspondences.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2),
               given N1 features from image 1, and N2 features from image 2.

        Returns:
            i2Fi1: Fundamental matrix, as 3x3 array.
            inlier_mask: boolean array of shape (N3,) indicating inlier matches.
        """
