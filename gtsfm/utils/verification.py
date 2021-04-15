"""Utilities for verification stage of the frontend.

Authors: Ayush Baid
"""
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Pose3, Rot3, Unit3

import gtsfm.utils.features as feature_utils
import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()


def recover_relative_pose_from_essential_matrix(
    i2Ei1: np.ndarray,
    verified_coordinates_i1: np.ndarray,
    verified_coordinates_i2: np.ndarray,
    camera_intrinsics_i1: Cal3Bundler,
    camera_intrinsics_i2: Cal3Bundler,
) -> Tuple[Rot3, Unit3]:
    """Recovers the relative rotation and translation direction from essential matrix and verified correspondences
    using opencv's API.

    Args:
        i2Ei1: essential matrix as a numpy array, of shape 3x3.
        verified_coordinates_i1: coordinates of verified correspondences in image i1, of shape Nx2.
        verified_coordinates_i2: coordinates of verified correspondences in image i2, of shape Nx2.
        camera_intrinsics_i1: intrinsics for image i1.
        camera_intrinsics_i2: intrinsics for image i2.

    Returns:
        relative rotation i2Ri1.
        relative translation direction i2Ui1.
    """
    # obtain points in normalized coordinates using intrinsics.
    normalized_coordinates_i1 = feature_utils.normalize_coordinates(verified_coordinates_i1, camera_intrinsics_i1)
    normalized_coordinates_i2 = feature_utils.normalize_coordinates(verified_coordinates_i2, camera_intrinsics_i2)

    # use opencv to recover pose
    _, i2Ri1, i2ti1, _ = cv.recoverPose(i2Ei1, normalized_coordinates_i1, normalized_coordinates_i2)
    i2Ri1 = Rot3(i2Ri1)
    i2Ui1 = Unit3(i2ti1.squeeze())
    i2Ei1_reconstructed = EssentialMatrix(i2Ri1, i2Ui1).matrix()

    # normalizing the two essential matrices
    i2Ei1_normalized = i2Ei1 / np.linalg.norm(i2Ei1, axis=None)
    i2Ei1_reconstructed_normalized = i2Ei1_reconstructed / np.linalg.norm(i2Ei1_reconstructed, axis=None)
    if not np.allclose(i2Ei1_normalized, i2Ei1_reconstructed_normalized):
        logger.warn("Recovered R, t cannot create the input Essential Matrix")

    return i2Ri1, i2Ui1


def fundamental_to_essential_matrix(
    i2Fi1: np.ndarray, camera_intrinsics_i1: Cal3Bundler, camera_intrinsics_i2: Cal3Bundler
) -> np.ndarray:
    """Converts the fundamental matrix to essential matrix using camera intrinsics.

    Args:
        i2Fi1: fundamental matrix which maps points in image #i1 to lines in image #i2.
        camera_intrinsics_i1: intrinsics for image #i1.
        camera_intrinsics_i2: intrinsics for image #i2.

    Returns:
        Estimated essential matrix i2Ei1 as numpy array of shape (3x3).
    """
    return camera_intrinsics_i2.K().T @ i2Fi1 @ camera_intrinsics_i1.K()


def essential_to_fundamental_matrix(
    i2Ei1: EssentialMatrix, camera_intrinsics_i1: Cal3Bundler, camera_intrinsics_i2: Cal3Bundler
) -> np.ndarray:
    """Converts the essential matrix to fundamental matrix using camera intrinsics.

    Args:
        i2Ei1: essential matrix which maps points in image #i1 to lines in image #i2.
        camera_intrinsics_i1: intrinsics for image #i1.
        camera_intrinsics_i2: intrinsics for image #i2.

    Returns:
        Fundamental matrix i2Fi1 as numpy array of shape (3x3).
    """
    return np.linalg.inv(camera_intrinsics_i2.K().T) @ i2Ei1.matrix() @ np.linalg.inv(camera_intrinsics_i1.K())


def compute_epipolar_distances(
    normalized_coords_i1: np.ndarray, normalized_coords_i2: np.ndarray, i2Ei1: EssentialMatrix
) -> Optional[np.ndarray]:
    """Compute symmetric point-line epipolar distances between normalized coordinates of correspondences.

    Args:
        normalized_coords_i1: normalized coordinates in image i1, of shape Nx2.
        normalized_coords_i2: corr. normalized coordinates in image i2, of shape Nx2.
        i2Ei1: essential matrix between two images

    Returns:
        Symmetric epipolar distances for each row of the input, of shape N.
    """
    if (
        normalized_coords_i1 is None
        or normalized_coords_i1.size == 0
        or normalized_coords_i2 is None
        or normalized_coords_i2.size == 0
    ):
        return None

    # construct the essential matrix in the opposite directin
    i2Ti1 = Pose3(i2Ei1.rotation(), i2Ei1.direction().point3())
    i1Ti2 = i2Ti1.inverse()
    i1Ei2 = EssentialMatrix(i1Ti2.rotation(), Unit3(i1Ti2.translation()))

    # get lines in i2 and i1
    epipolar_lines_i2 = feature_utils.convert_to_epipolar_lines(normalized_coords_i1, i2Ei1)
    epipolar_lines_i1 = feature_utils.convert_to_epipolar_lines(normalized_coords_i2, i1Ei2)

    # compute two distances and average them
    return 0.5 * (
        feature_utils.compute_point_line_distances(normalized_coords_i1, epipolar_lines_i1)
        + feature_utils.compute_point_line_distances(normalized_coords_i2, epipolar_lines_i2)
    )
