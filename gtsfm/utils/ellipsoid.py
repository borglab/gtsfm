"""Algorithms to center and align 3D points and camera frustums to the x, y, and z axes using SVD. Used in React
Three Fiber Visulization Tool.

Authors: Adi Singh
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple
from gtsam import Pose3, Rot3

from gtsfm.common.gtsfm_data import GtsfmData


def transform_point_cloud_wrapper(gtsfm_data: GtsfmData) -> Pose3:
    """Wrapper function for all the functions in ellipsoid.py. Transforms the point cloud contained within gtsfm_Data
    to be aligned with the x, y, and z axes.

    Args:
        gtsfm_data: scene data to write to transform.

    Returns:
        The final transformation required to align point cloud and frustums.
    """
    # Iterate through each track to gather a list of 3D points forming the point cloud.
    point_cloud_list = []
    num_pts = gtsfm_data.number_tracks()

    for j in range(num_pts):
        track = gtsfm_data.get_track(j)  # TEMP: might need an extra import for track
        x, y, z = track.point3()
        point_cloud_list.append([x, y, z])

    # Transform the point cloud to be aligned with x,y,z axes using SVD.
    point_cloud = np.array([np.array(p) for p in point_cloud_list])  # point_cloud has shape Nx3
    points_centered, mean = center_point_cloud(point_cloud)
    points_filtered = remove_outlier_points(points_centered)
    wuprightRw = get_alignment_rotation_matrix_from_svd(points_filtered)

    # Obtain the Pose3 object needed to align camera frustums.
    walignedTw = Pose3(Rot3(wuprightRw), -1 * mean)

    return walignedTw


def center_point_cloud(point_cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centers a point cloud using mean values of x, y, and z.

    Args:
        point_cloud: array of shape (N,3) representing the original point cloud.

    Returns:
        points_centered: array of shape (N,3) representing the centered point cloud
        mean: array of shape (3.) representing the mean x,y,z coordinates of point cloud

    Raises:
        TypeError: if point_cloud is not of shape (N,3).
    """
    if point_cloud.shape[1] != 3:
        raise TypeError("Points list should be 3D")

    mean = np.mean(point_cloud, axis=0)
    points_centered = point_cloud - mean
    return points_centered, mean


def remove_outlier_points(point_cloud: np.ndarray) -> np.ndarray:
    """Removes the top 5% of points with greatest distance from origin.

    Args:
        point_cloud: point cloud of shape N x 3.

    Returns:
        The filtered point cloud of shape M x 3 (M = 0.95*N).

    Raises:
        TypeError: if centered point cloud is not of shape (N,3).
    """
    if point_cloud.shape[1] != 3:
        raise TypeError("Point Cloud should be 3 dimensional")

    mags = np.linalg.norm(point_cloud, axis=1)
    cutoff_mag = np.percentile(mags, 95)
    points_filtered = point_cloud[mags < cutoff_mag]
    return points_filtered


def get_alignment_rotation_matrix_from_svd(point_cloud: np.ndarray) -> np.ndarray:
    """Applies SVD on a point cloud. The resulting V contains the rotation matrix required to align the 3 principal
    axes of the point cloud's ellipsoid with the x, y, z coordinate axes.

    Args:
        point_cloud: point cloud of shape (N,3).

    Returns:
        The rotation matrix, shape (3,3), required to align points with the x, y, and z axes.

    Raises:
        TypeError: if point cloud is not of shape (N,3).
    """
    if point_cloud.shape[1] != 3:
        raise TypeError("Point Cloud should be 3 dimensional")

    # U is NxN, S has 3 values along the diagonal, and Vt is 3x3
    # S represents the scaling (lengths) of each of the three ellipsoid semi-axes
    # ref: https://en.wikipedia.org/wiki/Singular_value_decomposition#Rotation,_coordinate_scaling,_and_reflection
    U, S, Vt = np.linalg.svd(point_cloud, full_matrices=True)
    wuprightRw = Vt
    return wuprightRw
