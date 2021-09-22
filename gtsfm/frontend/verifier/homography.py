from typing import Tuple

import cv2
import numpy as np

from gtsfm.common.keypoints import Keypoints

# from gtsfm.frontend.verifier.verifier_base import TwoViewEstimationReport

import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()

"""
Verification that determines whether the scene is planar or the camera motion is a pure rotation.

COLMAP also checks degeneracy of structure here:
    https://github.com/colmap/colmap/blob/dev/src/estimators/two_view_geometry.cc#L277
"""

MIN_PTS_HOMOGRAPHY = 4
DEFAULT_RANSAC_PROB = 0.999


class HomographyEstimator:
    def __init__(self, estimation_threshold_px: float) -> None:
        """
        """
        self._px_threshold = estimation_threshold_px

    def estimate(
        self, keypoints_i1: Keypoints, keypoints_i2: Keypoints, match_indices: np.ndarray
    ) -> Tuple[np.ndarray, float, int]:
        """Estimate to what extent the correspondences agree with an estimated homography.

        We provide statistics of the RANSAC result, like COLMAP does here for LORANSAC:
        https://github.com/colmap/colmap/blob/dev/src/optim/loransac.h

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2).

        Returns:
            H: array of shape (3,3) representing homography matrix.
            inlier_ratio: i.e. ratio of correspondences which approximately agree with planar geometry.
            num_inliers: number of correspondence consistent with estimated homography H.
        """
        if match_indices.shape[0] < MIN_PTS_HOMOGRAPHY:
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
            ransacReprojThreshold=self._px_threshold,
            # maxIters=10000,
            confidence=DEFAULT_RANSAC_PROB,
        )

        inlier_idxs = np.where(inlier_mask.ravel() == 1)[0]
        inlier_ratio = inlier_mask.mean()

        num_inliers = inlier_mask.sum()
        return H, num_inliers, inlier_ratio
