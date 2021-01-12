"""Common utilities for feature points.

Authors: Ayush Baid
"""
from typing import List

import cv2 as cv
import numpy as np
from gtsam import Cal3Bundler

from gtsfm.common.keypoints import Keypoints


def cast_to_gtsfm_keypoints(keypoints: List[cv.KeyPoint]) -> Keypoints:
    """Cast list of OpenCV's keypoints to GTSFM's keypoints.

    Args:
        keypoints: list of OpenCV's keypoints.

    Returns:
        GTSFM's keypoints with the same information as input keypoints.
    """
    coordinates = []
    scales = []
    responses = []
    for kp in keypoints:
        coordinates.append([kp.pt[0], kp.pt[1]])
        scales.append(kp.size)
        responses.append(kp.response)

    return Keypoints(
        coordinates=np.array(coordinates),
        scales=np.array(scales) if scales else None,
        responses=np.array(responses) if responses else None,
    )


def normalize_coordinates(
    coordinates: np.ndarray, intrinsics: Cal3Bundler
) -> np.ndarray:
    """Normalize 2D coordinates using camera intrinsics.

    Args:
        coordinates: 2d coordinates, of shape Nx2.
        intrinsics. camera intrinsics.

    Returns:
        normalized coordinates, of shape Nx2.
    """

    return np.vstack(
        [intrinsics.calibrate(x[:2].reshape(2, 1)) for x in coordinates]
    )
