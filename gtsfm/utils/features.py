"""Common utilities for feature points.

Authors: Ayush Baid
"""

from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.common.types import CALIBRATION_TYPE

EPS = 1e-8


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


def normalize_coordinates(coordinates: np.ndarray, intrinsics: CALIBRATION_TYPE) -> np.ndarray:
    """Normalize 2D coordinates using camera intrinsics.

    Args:
        coordinates: 2d coordinates, of shape Nx2.
        intrinsics. camera intrinsics.

    Returns:
        normalized coordinates, of shape Nx2.
    """
    return np.vstack([intrinsics.calibrate(x[:2].reshape(2, 1)) for x in coordinates])


def convert_to_homogenous_coordinates(non_homogenous_coordinates: np.ndarray) -> Optional[np.ndarray]:
    """Convert coordinates to homogenous system (by appending a column of ones).

    Args:
        non_homogenous_coordinates: 2d non-homogenous coordinates, of shape Nx2.

    Returns:
        2d homogenous coordinates, of shape Nx3.

    Raises:
        TypeError: if input is not 2 dimensional.
    """
    if non_homogenous_coordinates is None or non_homogenous_coordinates.size == 0:
        return None

    if non_homogenous_coordinates.shape[1] != 2:
        raise TypeError("Input should be 2D")

    return np.hstack((non_homogenous_coordinates, np.ones((non_homogenous_coordinates.shape[0], 1))))


def convert_to_epipolar_lines(coordinates_i1: np.ndarray, i2Fi1: np.ndarray) -> Optional[np.ndarray]:
    """Convert coordinates to epipolar lines in image i2.

    The epipolar line in image i2 is given by i2Fi1 @ x_i1. A point x_i2 is on this line if x_i2^T @ i2Fi1 @ x_i1 = 0.

    Args:
        coordinates_i1: coordinates in i1, of shape Nx2.
        i2Fi1: fundamental matrix.

    Returns:
        Corr. epipolar lines in i2, of shape Nx3.
    """
    if coordinates_i1 is None or coordinates_i1.size == 0:
        return None

    epipolar_lines = convert_to_homogenous_coordinates(coordinates_i1) @ i2Fi1.T
    return epipolar_lines


def point_line_dotproduct(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Computes the dot product of a point and a line in 2D. The function processed multiple inputs independently in a
    vectorized fashion.

    Note: the reason to not compute actual point-line distances is to flexible in providing different choices of
    denominator in the distance metric (e.g. SED, Sampson).

    Args:
        points: non-homogenous 2D points, of shape Nx2.
        lines: coefficients (a, b, c) of lines ax + by + c = 0, of shape Nx3.

    Returns:
        Point-line dot-product for each row, of shape N.
    """
    return np.sum(np.multiply(convert_to_homogenous_coordinates(points), lines), axis=1)


def generate_random_keypoints(num_keypoints: int, image_shape: Tuple[int, int]) -> Keypoints:
    """Generates random keypoints within the image bounds.

    Args:
        num_keypoints: Number of features to generate.
        image_shape: Size of the image, as (H,W)

    Returns:
        Generated keypoints.
    """
    if num_keypoints == 0:
        return Keypoints(coordinates=np.array([]))

    H, W = image_shape
    return Keypoints(coordinates=np.random.randint([0, 0], high=(W, H), size=(num_keypoints, 2)).astype(np.float32))
