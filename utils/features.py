"""Common utilities for feature points.

Authors: Ayush Baid
"""
from typing import List

import cv2 as cv
import numpy as np

from common.keypoints import Keypoints


def cast_to_gtsfm_keypoints(keypoints: List[cv.KeyPoint]) -> Keypoints:
    """Cast list of OpenCV's keypoints to GTSFM's keypoints.

    Args:
        keypoints: list of OpenCV's keypoints.

    Returns:
        GTSFM's keypoints with the same information as input keypoints.
    """

    data = [[kp.pt[0], kp.pt[1], kp.size, kp.response] for kp in keypoints]

    data = np.array(data, dtype=np.float32)

    return Keypoints(coordinates=data[:, :2],
                     scales=data[:, 2],
                     responses=data[:, 3])
