"""OpenCV RANSAC-based verifier implementation.

The verifier is the 5-Pt/8-pt Algorithm with RANSAC and is implemented by wrapping over 3rd party implementation.

References: 
- David Nistér. An efficient solution to the five-point relative pose problem. TPAMI, 2004.
- https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga13f7e34de8fa516a686a56af1196247f
- https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gae420abc34eaa03d0c6a67359609d8429

Authors: John Lambert
"""
from enum import Enum, unique
from typing import Tuple

import cv2
import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.verifier.opencv_verifier_base import OpencvVerifierBase

RANSAC_SUCCESS_PROB = 0.999999
RANSAC_MAX_ITERS = 1000000

logger = logger_utils.get_logger()


@unique
class RobustEstimationType(str, Enum):
    """Robust estimation algorithm types for OpenCV.

    See https://docs.opencv.org/4.x/d1/df1/md__build_master-contrib_docs-lin64_opencv_doc_tutorials_calib3d_usac.html
    for more detailed information. Note: USAC_FAST uses the RANSAC score to maximize number of inliers and terminate
    earlier.
    """

    FM_7POINT: str = "FM_7POINT"
    FM_8POINT: str = "FM_8POINT"
    FM_RANSAC: str = "FM_RANSAC"  # RANSAC algorithm. It needs at least 15 points. 7-point algorithm is used.
    RANSAC: str = "RANSAC"
    RHO: str = "RHO"
    USAC_DEFAULT: str = "USAC_DEFAULT"  # standard LO-RANSAC.
    USAC_PARALLEL: str = "USAC_PARALLEL"  # LO-RANSAC and RANSACs run in parallel.
    USAC_FM_8PTS: str = "USAC_FM_8PTS"  # LO-RANSAC. Only valid for Fundamental matrix with 8-points solver.
    USAC_FAST: str = "USAC_FAST"  # LO-RANSAC with smaller number iterations in local optimization step.
    USAC_ACCURATE: str = "USAC_ACCURATE"  # GC-RANSAC.
    USAC_PROSAC: str = "USAC_PROSAC"  # PROSAC sampling. Note, points must be sorted.
    USAC_MAGSAC: str = "USAC_MAGSAC"  # MAGSAC++.


class Ransac(OpencvVerifierBase):
    def estimate_E(
        self,
        uv_norm_i1: np.ndarray,
        uv_norm_i2: np.ndarray,
        match_indices: np.ndarray,
        fx: float,
        robust_estimation_type: RobustEstimationType = RobustEstimationType.USAC_ACCURATE,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the Essential matrix from correspondences.

        Args:
            uv_norm_i1: Normalized coordinates of detected features in image #i1.
            uv_norm_i2: Normalized coordinates of detected features in image #i2.
            match_indices: Matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2),
               given N1 features from image 1, and N2 features from image 2.
            fx: Focal length (in pixels) in horizontal direction.

        Returns:
            i2Ei1: Essential matrix, as 3x3 array.
            inlier_mask: boolean array of shape (N3,) indicating inlier matches.
        """
        K = np.eye(3)
        i2Ei1, inlier_mask = cv2.findEssentialMat(
            uv_norm_i1[match_indices[:, 0]],
            uv_norm_i2[match_indices[:, 1]],
            K,
            method=getattr(cv2, robust_estimation_type.value),
            threshold=self._estimation_threshold_px / fx,
            prob=RANSAC_SUCCESS_PROB,
        )
        return i2Ei1, inlier_mask

    def estimate_F(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        robust_estimation_type: RobustEstimationType = RobustEstimationType.FM_RANSAC,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the Fundamental matrix from correspondences.

        Args:
            keypoints_i1: Detected features in image #i1.
            keypoints_i2: Detected features in image #i2.
            match_indices: Matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2),
               given N1 features from image 1, and N2 features from image 2.

        Returns:
            i2Fi1: Fundamental matrix, as 3x3 array.
            inlier_mask: Boolean array of shape (N3,) indicating inlier matches.
        """
        i2Fi1, inlier_mask = cv2.findFundamentalMat(
            keypoints_i1.extract_indices(match_indices[:, 0]).coordinates,
            keypoints_i2.extract_indices(match_indices[:, 1]).coordinates,
            method=getattr(cv2, robust_estimation_type.value),
            ransacReprojThreshold=self._estimation_threshold_px,
            confidence=RANSAC_SUCCESS_PROB,
            maxIters=RANSAC_MAX_ITERS,
        )
        return i2Fi1, inlier_mask
