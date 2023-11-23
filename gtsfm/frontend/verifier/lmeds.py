"""Least Median of Squares (LMEDS) verifier.

See Peter J. Rousseeuw, "Least Median of Squares Regression", 1984.
https://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/LeastMedianOfSquares.pdf

Authors: Ayush Baid, John Lambert
"""

from typing import Tuple

import cv2
import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.verifier.opencv_verifier_base import OpencvVerifierBase


class LMEDS(OpencvVerifierBase):
    def estimate_E(
        self, uv_norm_i1: np.ndarray, uv_norm_i2: np.ndarray, match_indices: np.ndarray, fx: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the Essential matrix from correspondences.

        Args:
            uv_norm_i1: normalized coordinates of detected features in image #i1.
            uv_norm_i2: normalized coordinates of detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2),
               given N1 features from image 1, and N2 features from image 2.
            fx: focal length (in pixels) in horizontal direction (unused).

        Returns:
            i2Ei1: Essential matrix, as 3x3 array.
            inlier_mask: boolean array of shape (N3,) indicating inlier matches.
        """
        K = np.eye(3)
        i2Ei1, inlier_mask = cv2.findEssentialMat(
            uv_norm_i1[match_indices[:, 0]],
            uv_norm_i2[match_indices[:, 1]],
            K,
            method=cv2.FM_LMEDS,
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
            method=cv2.FM_LMEDS,
        )
        return i2Fi1, inlier_mask
