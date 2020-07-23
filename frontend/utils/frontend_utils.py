"""
Utility function for front-end results.

Authors: Ayush Baid, John Lambert
"""
from typing import Tuple

import cv2 as cv
import gtsam
import numpy as np


def essential_matrix_from_fundamental_matrix(
        fundamental_matrix: np.ndarray,
        intrinsics_im1: np.ndarray,
        intrinsics_im2: np.ndarray) -> np.ndarray:
    """Compute the essential matrix from fundamental matrix and camera intrinsics using
    simple matrix multiplication.

    Args:
        fundamental_matrix (np.ndarray): fundamental matrix from im1 to im2 (3x3).
        intrinsics_im1 (np.ndarray): camera calibration for im1 (3x3).
        intrinsics_im2 (np.ndarray): camera calibration for im2 (3x3).

    Returns:
        np.ndarray: essential matrix (3x3).
    """
    return intrinsics_im2.T @ fundamental_matrix @ intrinsics_im1


def decompose_essential_matrix(
    essential_matrix: np.ndarray,
    points_im1: np.ndarray,
    points_im2: np.ndarray
) -> Tuple[gtsam.Rot3,gtsam.Unit3]:
    """Decompose essential matrix into the relative pose between the pair of cameras using OpenCV.

    Args:
        essential_matrix (np.ndarray): essential_matrix (3x3)
        points_im1 (np.ndarray): geometrically verified points from im1 which match to points in im2.
        points_im2 (np.ndarray): geometrically verified points from im2 which correspond to points_im1,

    Returns:
        gtsam.Pose3: relative pose from im1 to im2.
    """
    _, rotation, translation, _ = cv.recoverPose(
        essential_matrix, points_im1[:, :2], points_im2[:, :2])

    # will automatically normalize "t" to unit length
    return gtsam.Rot3(rotation), gtsam.Unit3(translation.flatten())
