""" 
Common utilities for feature points.

Authors: Ayush Baid
"""
from typing import List

import cv2 as cv
import numpy as np


def convert_to_opencv_keypoints(features: np.ndarray) -> List[cv.KeyPoint]:
    """
    Converts the features from numpy array to cv keypoints.

    Args:
        features (np.ndarray): features as numpy array

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
