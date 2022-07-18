"""
RANSAC verifier implementation.

The verifier is the 5-Pt/8-pt Algorithm with RANSAC and is implemented by wrapping over 3rd party implementation.

References: 
- David Nistér. An efficient solution to the five-point relative pose problem. TPAMI, 2004.
- https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga13f7e34de8fa516a686a56af1196247f
- https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gae420abc34eaa03d0c6a67359609d8429

Authors: John Lambert
"""

from typing import Tuple

import cv2
import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.verifier.opencv_verifier_base import OpencvVerifierBase


RANSAC_SUCCESS_PROB = 0.999999
RANSAC_MAX_ITERS = 1000000

logger = logger_utils.get_logger()

"""Alternative options:

  cv2.RANSAC
  cv2.RHO
  cv2.USAC_DEFAULT
  cv2.USAC_PARALLEL
  cv2.USAC_FM_8PTS
  cv2.USAC_FAST
  cv2.USAC_ACCURATE
  cv2.USAC_PROSAC
  cv2.USAC_MAGSAC
"""


class Ransac(OpencvVerifierBase):
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
        K = np.eye(3)
        i2Ei1, inlier_mask = cv2.findEssentialMat(
            uv_norm_i1[match_indices[:, 0]],
            uv_norm_i2[match_indices[:, 1]],
            K,
            method=cv2.USAC_ACCURATE,
            threshold=self._estimation_threshold_px / fx,
            prob=RANSAC_SUCCESS_PROB,
        )
        return i2Ei1, inlier_mask

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
        i2Fi1, inlier_mask = cv2.findFundamentalMat(
            keypoints_i1.extract_indices(match_indices[:, 0]).coordinates,
            keypoints_i2.extract_indices(match_indices[:, 1]).coordinates,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=self._estimation_threshold_px,
            confidence=RANSAC_SUCCESS_PROB,
            maxIters=RANSAC_MAX_ITERS,
        )
        return i2Fi1, inlier_mask
