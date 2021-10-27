
"""
Fit a homography matrix from correspondences.

Useful for determining whether the scene is planar or the camera motion is a pure rotation.

COLMAP also checks degeneracy of structure here:
    https://github.com/colmap/colmap/blob/dev/src/estimators/two_view_geometry.cc#L277

Authors: John Lambert
"""

from typing import Tuple

import cv2
import numpy as np

import gtsfm.utils.logger as logger_utils
import gtsfm.frontend.homography_verifier.homography_verifier_base as homography_verifier_base
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.homography_verifier.homography_verifier_base import HomographyVerifierBase

logger = logger_utils.get_logger()


DEFAULT_RANSAC_PROB = 0.999


class RansacHomographyVerifier(HomographyVerifierBase):

    def verify(
        self, keypoints_i1: Keypoints, keypoints_i2: Keypoints, match_indices: np.ndarray, estimation_threshold_px: float
    ) -> Tuple[np.ndarray, float, int, np.ndarray]:
        """Verify that a set of correspondences belong to a homography configuration.

        We fit a homography to the correspondences, and also estimate to what extent the correspondences agree
        with the estimated homography.

        We provide statistics of the RANSAC result, like COLMAP does here for LORANSAC:
        https://github.com/colmap/colmap/blob/dev/src/optim/loransac.h

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2).
            estimation_threshold_px: threshold value (in pixels) to use for classifying inliers in RANSAC.

       Returns:
            H: array of shape (3,3) representing homography matrix.
            inlier_idxs: indices of inliers from matches array.
            inlier_ratio: i.e. ratio of correspondences which approximately agree with planar geometry.
            num_inliers: number of correspondence consistent with estimated homography H.
        """
        if match_indices.shape[0] < homography_verifier_base.MIN_PTS_HOMOGRAPHY:
            num_inliers = 0
            inlier_ratio = 0.0
            return num_inliers, inlier_ratio

        uv_i1 = keypoints_i1.coordinates
        uv_i2 = keypoints_i2.coordinates

        # TODO(johnwlambert): cast as np.float32?
        H, inlier_mask = cv2.findHomography(
            srcPoints=uv_i1[match_indices[:, 0]],
            dstPoints=uv_i2[match_indices[:, 1]],
            method=cv2.RANSAC,
            ransacReprojThreshold=estimation_threshold_px,
            # maxIters=10000,
            confidence=DEFAULT_RANSAC_PROB,
        )

        inlier_idxs = np.where(inlier_mask.ravel() == 1)[0]
        inlier_ratio = inlier_mask.mean()

        num_inliers = inlier_mask.sum()
        return H, inlier_idxs, num_inliers, inlier_ratio
