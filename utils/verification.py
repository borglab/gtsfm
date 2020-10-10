from typing import Tuple, Union

import cv2 as cv
import gtsam
import numpy as np
from gtsam import Rot3, Unit3

import utils.features as feature_utils


def generate_virtual_correspondence(im1_size, im2_size, f_gt, num_points_eval=None):
    pts1_grid, pts2_grid = feature_utils.generate_features_grid(
        im1_size, im2_size)

    if num_points_eval is None:
        num_points_eval = pts1_grid.shape[0]

    pts1_virt, pts2_virt = cv.correctMatches(
        f_gt, np.reshape(pts1_grid, (1, -1, 2)),
        np.reshape(pts2_grid, (1, -1, 2))
    )

    valid_1 = np.logical_and(
        np.logical_not(np.isnan(pts1_virt[:, :, 0])),
        np.logical_not(np.isnan(pts1_virt[:, :, 1])),
    )
    valid_2 = np.logical_and(
        np.logical_not(np.isnan(pts2_virt[:, :, 0])),
        np.logical_not(np.isnan(pts2_virt[:, :, 1])),
    )

    valid_idx = np.where(np.logical_and(valid_1, valid_2))

    if len(valid_idx) < num_points_eval:
        return pts1_virt[0, :, :], pts2_virt[0, :, :]

    return pts1_virt[0, :num_points_eval, :], pts2_virt[0, :num_points_eval, :]


def normalize_matrix(f_matrix: np.array,
                     norm_mode: str) -> np.array:
    '''
    Normalize the fundamental matrix with either frobenious norm or inf norm

    TODO: write a test

    Args:
        f_matrix (numpy array):     fundamental matrix
        norm_mode (str):            either "frobenious" or "maxabs"; the type of normalization to perform
    Returns:
        numpy array: the normalized fundamental matrix
    '''

    eps = 1e-10

    if norm_mode == 'frobenious':
        return f_matrix/(np.linalg.norm(f_matrix, ord='fro')+eps)
    elif norm_mode == 'maxabs':
        return f_matrix/(np.max(np.abs(f_matrix), axis=None)+eps)

    raise AttributeError('Invalid normalization mode')


def generate_essential_matrix(rotation, translation):
    # TODO: write unit test

    tx = np.array([
        [0, -translation[2], translation[1]],
        [translation[2], 0, -translation[0]],
        [-translation[1], translation[0], 0]
    ])

    return np.matmul(rotation, tx)


def generate_fundamental_matrix(instrinsic_1, instrinsic_2, essential_mat):
    # TODO: write unit test

    return np.matmul(np.linalg.inv(np.transpose(instrinsic_2)),
                     np.matmul(essential_mat, np.linalg.inv(instrinsic_1)))


def normalize_coordinates(features: np.ndarray,
                          intrinsics_mat: np.ndarray) -> np.ndarray:
    """Normalizes the feature coordinates using the camera intrinsics

    Args:
        features (np.ndarray): Homogenous feature coordinates in pixel
                               coordinates. Shape=[Nx3].
        intrinsics_mat (np.ndarray): 3x3 camera intrinsics matrix.

    Raises:
        ValueError: incorrect dimensions of input features

    Returns:
        np.ndarray: normalized features
    """

    if features.shape[1] == 2:
        temp = feature_utils.convert_to_homogenous(features) @ \
            np.linalg.inv(intrinsics_mat)
        temp[:, 0] /= temp[:, 2]
        temp[:, 1] /= temp[:, 2]

        return temp[:, :1]
    elif features.shape[1] == 3:
        return features @ np.linalg.inv(intrinsics_mat)

    raise ValueError('Incorrect dimension of input features')


def intrinsics_from_image_shape(image_shape: Tuple[int, int]) -> np.ndarray:
    """Approximate intrinsics from image shape.

    Args:
        image_shape (Tuple[int, int]): shape of the image

    Returns:
        np.ndarray: approximated intrinsics
    """
    return np.array([
        [image_shape[0], 0, image_shape[0]/2],
        [0, image_shape[1], image_shape[1]/2],
        [0, 0, 1]
    ])


def relative_pose_from_fundamental_matrix(f_matrix: np.ndarray,
                                          K1: np.ndarray,
                                          K2: np.ndarray,
                                          points1: np.ndarray,
                                          points2: np.ndarray) -> gtsam.Pose3:
    e_matrix = K2.T @ f_matrix @ K1

    # recover the pose using opencv
    _, R, t, _ = cv.recoverPose(e_matrix, points1[:, :2], points2[:, :2])

    return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t.flatten()))


def recover_pose_from_fundamental_matrix(
    frontend_output: Tuple[np.ndarray, np.ndarray, np.ndarray],
    camera_intrinsics_im1: np.ndarray,
    camera_intrinsics_im2: np.ndarray,
) -> Tuple[Union[Rot3, None], Union[Unit3, None]]:
    """Recover pose from output of the frontend.

    Args:
        frontend_output: tuple of F matrix and verified correspondences from
                         frontend.
        camera_intrinsics_im1: intrinsics for im1.
        camera_intrinsics_im2: intrinsics for im2.

    Returns:
        Union[Rot3, None]: recovered rotation from im1 to im2.
        Union[Unit3, None]: recovered unit translation from im1 to im2.
    """

    if frontend_output[0] is None or frontend_output[1] is None:
        return None

    if frontend_output[0].size == 0 or frontend_output[1].size == 0:
        return None

    e_matrix = camera_intrinsics_im2.T @ frontend_output[0] @ \
        camera_intrinsics_im1

    coords_im1 = normalize_coordinates(
        frontend_output[1][:, :2], camera_intrinsics_im1)

    coords_im2 = normalize_coordinates(
        frontend_output[2][:, :2], camera_intrinsics_im2)

    # recover the pose using opencv
    _, R, t, _ = cv.recoverPose(
        e_matrix,
        coords_im1,
        coords_im2,
        np.eye(3))

    return Rot3(R), Unit3(t)


def recover_pose_from_essential_matrix(
    frontend_output: Tuple[np.ndarray, np.ndarray, np.ndarray],
    camera_intrinsics_im1: np.ndarray,
    camera_intrinsics_im2: np.ndarray,
) -> Tuple[Union[Rot3, None], Union[Unit3, None]]:
    """Recover pose from output of the frontend.

    Args:
        frontend_output: tuple of E matrix and verified correspondences from
                         frontend.
        camera_intrinsics_im1: intrinsics for im1.
        camera_intrinsics_im2: intrinsics for im2.

    Returns:
        Union[Rot3, None]: recovered rotation from im1 to im2.
        Union[Unit3, None]: recovered unit translation from im1 to im2.
    """

    if frontend_output[0] is None or frontend_output[1] is None:
        return None

    if frontend_output[0].size == 0 or frontend_output[1].size == 0:
        return None

    e_matrix = frontend_output[0]

    coords_im1 = normalize_coordinates(
        frontend_output[1][:, :2], camera_intrinsics_im1)

    coords_im2 = normalize_coordinates(
        frontend_output[2][:, :2], camera_intrinsics_im2)

    # recover the pose using opencv
    _, R, t, _ = cv.recoverPose(
        e_matrix,
        coords_im1,
        coords_im2,
        np.eye(3))

    return Rot3(R), Unit3(t)
