"""Utilities for verification stage of the frontend.

Authors: Ayush Baid
"""
from typing import Tuple

import cv2 as cv
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Pose3, Rot3, Unit3

import utils.features as feature_utils


def recover_relative_pose_from_essential_matrix(
        i2Ei1: np.ndarray,
        verified_coordinates_i1: np.ndarray,
        verified_coordinates_i2: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler) -> Tuple[Rot3, Unit3]:
    """Recovers the relative rotation and translation direction from essential
    matrix and verified correspondences using opencv's API.

    Args:
        i2Ei1: essential matrix as a numpy array, of shape 3x3.
        verified_coordinates_i1: coordinates of verified correspondences in
                                 image i1, of shape Nx2.
        verified_coordinates_i2: coordinates of verified correspondences in
                                 image i2, of shape Nx2.
        camera_intrinsics_i1: intrinsics for image i1.
        camera_intrinsics_i2: intrinsics for image i2.

    Returns:
        relative rotation i2Ri1.
        relative translation direction i2Ui1.
    """
    # obtain points in normalized coordinates using intrinsics.
    normalized_coordinates_i1 = feature_utils.normalize_coordinates(
        verified_coordinates_i1, camera_intrinsics_i1)
    normalized_coordinates_i2 = feature_utils.normalize_coordinates(
        verified_coordinates_i2, camera_intrinsics_i2)

    # use opencv to recover pose
    _, i2_R_i1, i2_t_i1, _ = cv.recoverPose(i2Ei1,
                                normalized_coordinates_i1,
                                normalized_coordinates_i2)

    return Rot3(i2_R_i1), Unit3(i2_t_i1.squeeze())


def create_essential_matrix(i2Ri1: Rot3, i2Ui1: Unit3) -> EssentialMatrix:
    """Creates essential matrix from rot3 and unit3.

    Args:
        i2Ri1: relative rotation.
        i2Ui1: relative translation direction.

    Returns:
        essential matrix i2Ei1.
    """
    return EssentialMatrix(i2Ri1, i2Ui1)


def fundamental_to_essential_matrix(i2Fi1: np.ndarray,
                                    camera_intrinsics_i1: Cal3Bundler,
                                    camera_intrinsics_i2: Cal3Bundler
                                    ) -> np.ndarray:
    """Converts the fundamental matrix to essential matrix using camera intrinsics.

    Args:
        i2Fi1: fundamental matrix which maps points in image #i1 to lines
               in image #i2.
        camera_intrinsics_i1: intrinsics for image #i1.
        camera_intrinsics_i2: intrinsics for image #i2.

    Returns:
            Estimated essential matrix i2Ei1 as numpy array of shape (3x3).
    """
    return camera_intrinsics_i2.K().T @ \
        i2Fi1 @ \
        camera_intrinsics_i1.K()
