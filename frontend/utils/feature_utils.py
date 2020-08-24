""" 
Common utilities for feature points.

Authors: Ayush Baid
"""
from typing import List

import cv2 as cv
import numpy as np


def keypoints_of_array(features: np.ndarray) -> List[cv.KeyPoint]:
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


def array_of_keypoints(keypoints: List[cv.KeyPoint]) -> np.ndarray:
    """
    Converts the cv keypoints to a numpy array, the standard feature representation in GTSFM

    Args:
        keypoints (List[cv.KeyPoint]): keypoints representation of the given features

    Returns:
        np.ndarray: features
    """

    if len(keypoints) == 0:
        return np.array([])

    response_scores = [kp.response for kp in keypoints]
    # adding an offset to prevent division by zero
    max_response = max(response_scores) + 1e-6
    min_response = min(response_scores)

    feat_list = [[kp.pt[0], kp.pt[1], kp.size, (kp.response-min_response)/(max_response-min_response)]
                 for kp in keypoints]

    return np.array(feat_list, dtype=np.float32)


def convert_to_epipolar_lines(coordinates: np.array,
                              f_matrix: np.array,
                              is_normalize: bool = False) -> np.array:
    '''
    Convert the feature coordinates to epipolar lines

    Args:
        coordinates (numpy array):  coordinates of the features
        f_matrix (numpy array):     fundamental matrix
        is_normalize (bool):        flag indicating if the epipolar lines should be normalized to unit norm
    Returns:
        bool numpy array: the boolean flag for each input
    '''
    if coordinates is None or coordinates.size == 0:
        return coordinates

    # convert the feature_coordinates_1 to epipolar lines
    epipolar_lines = np.matmul(convert_to_homogenous(coordinates),
                               np.transpose(f_matrix)
                               )

    # normalize the lines
    if is_normalize:
        lines_norm = np.linalg.norm(epipolar_lines, axis=1, keepdims=True)

        epipolar_lines = np.divide(epipolar_lines, lines_norm)

    return epipolar_lines


def convert_to_homogenous(normal_coordinates: np.array) -> np.array:
    '''
    Convert normal coordinates to homogenous coordinates

    Args:
        normal_coordinates (numpy array): Normal (x,y) 2d point coordinates with shape Nx2
    Returns
        numpy array: the corresponding homogenous coordinates. Shape=Nx3
    '''

    if np.shape(normal_coordinates)[1] == 3:
        return normal_coordinates

    return np.append(normal_coordinates, np.ones((normal_coordinates.shape[0], 1)), axis=1)


def generate_pointlinedistance_vector_epipolar(feature_coords_1: np.array,
                                               feature_coords_2: np.array,
                                               f_matrix: np.array,
                                               mode: str = 'single') -> np.array:
    '''
    Get the point-line epipolar between the features under epipolar transform (row-level correspondence)

    Args:
        feature_coords_1 (numpy array): coordinates from the 1st image
        feature_coords_2 (numpy array): coordinates from the 2nd image
        f_matrix (numpy array):         fundamental matrix 
        mode (str):                     'single' (one way distances) v 'double' (two way distances)
    Returns:
        numpy array: distance computation for each row of the input coordinates
    '''
    if feature_coords_1 is None or feature_coords_1.shape[0] == 0 or feature_coords_2 is None or feature_coords_2.shape[0] == 0:
        return np.array([])

    if feature_coords_1.shape[1] == 2:
        # convert to homogenous coordinates
        feature_coords_1 = convert_to_homogenous(feature_coords_1)

    if feature_coords_2.shape[1] == 2:
        # convert to homogenous coordinates
        feature_coords_2 = convert_to_homogenous(feature_coords_2)

    epipolar_lines_1 = convert_to_epipolar_lines(feature_coords_1, f_matrix)
    # normalizing the lines for the first two columns
    epipolar_lines_1 = np.divide(epipolar_lines_1, np.linalg.norm(
        epipolar_lines_1[:, :2], axis=1, keepdims=True))

    left_dist = np.abs(
        np.sum(np.multiply(epipolar_lines_1, feature_coords_2), axis=1)
    )

    if mode == 'double':
        right_dist = generate_pointlinedistance_vector_epipolar(
            feature_coords_2,
            feature_coords_1,
            np.transpose(f_matrix),
            mode='single'
        )

        return 0.5*(left_dist+right_dist)
    else:
        return left_dist
