"""Common utilities for feature points.

Authors: Ayush Baid
"""
from typing import List

import cv2 as cv
import numpy as np


def keypoints_from_array(features: np.ndarray) -> List[cv.KeyPoint]:
    """Converts the features from numpy array to cv keypoints.

    Args:
        features: Numpy array of shape (N,2+) representing feature points.

    Returns:
        OpenCV KeyPoint objects representing the given features.
    """
    # TODO(ayush): what should be scale if not provided?

    # input features is a 2D array
    if features.shape[1] < 3:
        keypoints = [cv.KeyPoint(
            x=float(f[0]),
            y=float(f[1]),
            _size=2) for f in features]
    elif features.shape[1] < 4:
        keypoints = [cv.KeyPoint(
            x=float(f[0]),
            y=float(f[1]),
            _size=float([2])) for f in features]
    else:
        keypoints = [cv.KeyPoint(
            x=float(f[0]),
            y=float(f[1]),
            _size=float(f[2]),
            _response=float(f[3])) for f in features]

    return keypoints


def array_from_keypoints(keypoints: List[cv.KeyPoint]) -> np.ndarray:
    """Converts the opencv keypoints to a numpy array, the standard feature
    representation in GTSFM.

    Args:
        keypoints: OpenCV's keypoint representation of the given features.

    Returns:
        Array of shape (N, 4) representing features.
    """

    feat_list = [[kp.pt[0], kp.pt[1], kp.size, kp.response] for kp in keypoints]

    return np.array(feat_list, dtype=np.float32)
