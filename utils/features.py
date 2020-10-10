"""Common utilities for feature points.

Authors: Ayush Baid
"""
from typing import List

import cv2 as cv
import numpy as np


def keypoints_of_array(features: np.ndarray) -> List[cv.KeyPoint]:
    """Converts the features from numpy array to cv keypoints.

    Args:
        features: features as numpy array

    Returns:
        List[cv.KeyPoint]: keypoints representation of the given features
    """
    # TODO(ayush): what should be scale if not provided?

    # input features is a 2D array
    if features.shape[1] < 3:
        keypoints = [cv.KeyPoint(x=f[0], y=f[1], _size=2) for f in features]
    else:
        keypoints = [cv.KeyPoint(x=f[0], y=f[1], _size=f[2]) for f in features]

    return keypoints


def array_of_keypoints(keypoints: List[cv.KeyPoint]) -> np.ndarray:
    """Converts the opencv keypoints to a numpy array, the standard feature 
    representation in GTSFM.

    Args:
        keypoints: OpenCV's keypoint representation of the given features

    Returns:
        np.ndarray: features
    """

    feat_list = [[kp.pt[0], kp.pt[1], kp.size, kp.response] for kp in keypoints]

    return np.array(feat_list, dtype=np.float32)
