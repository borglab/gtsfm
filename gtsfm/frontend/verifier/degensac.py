"""
Locally Optimized (LO) Degensac verifier implementation.

The verifier is a combination of 'Locally Optimized Ransac' and 'Two-view Geometry Estimation Unaffected by a Dominant
Plane' and is implemented by wrapping over 3rd party implementation.

References:
- https://link.springer.com/chapter/10.1007/978-3-540-45243-0_31
- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.466.2719&rep=rep1&type=pdf
- https://github.com/ducha-aiki/pyransac

Authors: Ayush Baid, John Lambert
"""

from typing import Optional, Tuple

import numpy as np
import pydegensac
from gtsam import Cal3Bundler, Rot3, Unit3

import gtsfm.utils.logger as logger_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.verifier.verifier_base import VerifierBase

logger = logger_utils.get_logger()


class Degensac(VerifierBase):
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
        Raises:
            NotImplementedError: when configured to compute essential matrices.
        """
        self._estimation_threshold_px = estimation_threshold_px
        if use_intrinsics_in_verification is True:
            raise NotImplementedError("DEGENSAC cannot estimate essential matrices")

        super().__init__(use_intrinsics_in_verification, estimation_threshold_px)

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
            Inlier ratio of w.r.t. the estimated model, i.e. the #final RANSAC inliers/ #putatives.
        """
        if match_indices.shape[0] < self._min_matches:
            return self._failure_result

        i2Fi1, inlier_mask = pydegensac.findFundamentalMatrix(
            keypoints_i1.coordinates[match_indices[:, 0]],
            keypoints_i2.coordinates[match_indices[:, 1]],
            px_th=self._estimation_threshold_px,
        )
        inlier_idxs = np.where(inlier_mask.ravel() == 1)[0]

        v_corr_idxs = match_indices[inlier_idxs]
        inlier_ratio_est_model = np.mean(inlier_mask)

        i2Ei1_matrix = verification_utils.fundamental_to_essential_matrix(
            i2Fi1, camera_intrinsics_i1, camera_intrinsics_i2
        )

        i2Ri1, i2Ui1 = verification_utils.recover_relative_pose_from_essential_matrix(
            i2Ei1_matrix,
            keypoints_i1.coordinates[match_indices[inlier_idxs, 0]],
            keypoints_i2.coordinates[match_indices[inlier_idxs, 1]],
            camera_intrinsics_i1,
            camera_intrinsics_i2,
        )

        return i2Ri1, i2Ui1, v_corr_idxs, inlier_ratio_est_model
