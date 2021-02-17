"""Common utilities for feature points.

Authors: Ayush Baid
"""
from typing import List

import cv2 as cv
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix

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


def normalize_coordinates(coordinates: np.ndarray, intrinsics: Cal3Bundler) -> np.ndarray:
    """Normalize 2D coordinates using camera intrinsics.

    Args:
        coordinates: 2d coordinates, of shape Nx2.
        intrinsics. camera intrinsics.

    Returns:
        normalized coordinates, of shape Nx2.
    """
    return np.vstack([intrinsics.calibrate(x[:2].reshape(2, 1)) for x in coordinates])


def convert_to_homogenous_coordinates(
    non_homogenous_coordinates: np.ndarray,
) -> np.ndarray:
    """Convert coordinates to homogenous system (by appending a column of ones).

    Args:
        non_homogenous_coordinates: 2d non-homogenous coordinates, of shape Nx2.

    Returns:
        2d homogenous coordinates, of shape Nx3.

    Raises:
        TypeError: if input is not 2 dimensional.
    """
    if non_homogenous_coordinates is None or non_homogenous_coordinates.size == 0:
        return non_homogenous_coordinates

    if non_homogenous_coordinates.shape[1] != 2:
        raise TypeError("Input should be 2D")

    return np.hstack(
        (
            non_homogenous_coordinates,
            np.ones((non_homogenous_coordinates.shape[0], 1)),
        )
    )


def convert_to_epipolar_lines(normalized_coordinates_i1: np.ndarray, i2Ei1: EssentialMatrix) -> np.array:
    """Convert coordinates to epipolar lines.

    Args:
        normalized_coordinates_i1: normalized coordinates in i1, of shape Nx2.
        i2Ei1: essential matrix.

    Returns:
        Corr. epipolar lines in i2, of shape Nx3
    """
    if normalized_coordinates_i1 is None or normalized_coordinates_i1.size == 0:
        return normalized_coordinates_i1

    epipolar_lines = convert_to_homogenous_coordinates(normalized_coordinates_i1) @ i2Ei1.matrix().T

    return epipolar_lines


def compute_point_line_distances(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Computes the distance of a point from a line in 2D. The function processed multiple inputs independently in a
    vectorized fashion.

    Args:
        points: non-homogenous 2D points, of shape Nx2.
        lines: coefficients (a, b, c) of lines ax + by + c = 0, of shape Nx3.

    Returns:
        Point-line distance for each row, of shape N.
    """
    line_norms = np.linalg.norm(lines[:, :2], axis=1)

    return np.abs(np.sum(np.multiply(convert_to_homogenous_coordinates(points), lines), axis=1)) / line_norms
