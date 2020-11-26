"""Utilities for verification stage of the frontend.

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Point3, Rot3, Unit3


def cast_essential_matrix_to_gtsam(im2_E_im1: np.ndarray,
                                   verified_keypoints_im1: np.ndarray,
                                   verified_keypoints_im2: np.ndarray,
                                   camera_intrinsics_im1: Cal3Bundler,
                                   camera_intrinsics_im2: Cal3Bundler
                                   ) -> EssentialMatrix:
    """Cast essential matrix from numpy matrix to gtsam type.

    Args:
        im2_E_im1: essential matrix as numpy matrix of shape 3x3.
        verified_keypoints_im1: keypoints from image #1 which form verified
                                correspondences, of shape (N, 2+).
        verified_keypoints_im2: keypoints from image #1 which form verified
                                correspondences, of shape (N, 2+).
        camera_intrinsics_im1: intrinsics for image #1.
        camera_intrinsics_im2: intrinsics for image #2.

    Returns:
        EssentialMatrix: [description]
    """
    # TODO(ayush): move it to GTSAM as a constructor.

    # obtain points in normalized coordinates using intrinsics.
    normalized_verified_keypoints_im1 = np.vstack(
        [camera_intrinsics_im1.calibrate(
            x[:2].astype(np.float64).reshape(2, 1)
        ) for x in verified_keypoints_im1]
    ).astype(np.float32)

    normalized_verified_keypoints_im2 = np.vstack(
        [camera_intrinsics_im2.calibrate(
            x[:2].astype(np.float64).reshape(2, 1)
        ) for x in verified_keypoints_im2]
    ).astype(np.float32)

    # use opencv to recover pose
    _, R, t, _ = cv.recoverPose(im2_E_im1,
                                normalized_verified_keypoints_im1,
                                normalized_verified_keypoints_im2)

    return EssentialMatrix(Rot3(R), Unit3(Point3(t.squeeze())))


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
    return cal3bundler_to_matrix(camera_intrinsics_i2).T @ \
        i2Fi1 @ \
        cal3bundler_to_matrix(camera_intrinsics_i1)


def cal3bundler_to_matrix(intrinsics: Cal3Bundler) -> np.ndarray:
    """Gets the matrix representation of the intrinsics.

    Args:
        intrinsics: intrisics as Cal3Bundler object.

    Returns:
        3x3 matrix representation of the intrisnics.
    """

    return np.array([
        [intrinsics.fx(), 0, intrinsics.px()],
        [0, intrinsics.fx(), intrinsics.py()],
        [0, 0, 1]
    ])
