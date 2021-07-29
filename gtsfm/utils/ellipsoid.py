"""Algorithms to center and align 3D points and camera frusums to the x, y, and z axes using SVD. Used in React
Three Fiber Visulization Tool.

Authors: Adi Singh
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple

from gtsfm.common.gtsfm_data import GtsfmData


def transform_point_cloud_wrapper(gtsfm_data: GtsfmData) -> np.ndarray:
    """Wrapper function for all the functions in ellipsoid.py. Transforms the point cloud contained within gtsfm_Data
    to be aligned with the x, y, and z axes.

    Args:
        gtsfm_data: scene data to write to transform.

    Returns:
        aligned_points: transformed, aligned point cloud of shape (N,3)
        mean: array of shape (3.) representing the mean x,y,z coordinates of point cloud
        walignedRw: array of shape (3,3) representing the rotation matrix to align point cloud with axes
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
    walignedRw = get_rotation_matrix(points_filtered)
    aligned_points = apply_ellipsoid_rotation(walignedRw, points_centered)

    return aligned_points, mean, walignedRw


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


def get_rotation_matrix(point_cloud: np.ndarray) -> np.ndarray:
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
    walignedRw = Vt
    return walignedRw


def apply_ellipsoid_rotation(walignedRw: np.ndarray, point_cloud: np.ndarray) -> np.ndarray:
    """Applies a rotation on the centered, filtered point cloud.

    Args:
        walignedRw: rotation matrix, shape (3,3).
        point_cloud: point cloud, shape (N,3).

    Returns:
        A transformed, aligned point cloud of shape (N,3).

    Raises:
        TypeError: if rot matrix isn't (3,3) or centered_pc is not of shape (3,3).
    """
    if walignedRw.shape[0] != 3 or walignedRw.shape[1] != 3:
        raise TypeError("Rotation Matrix should be of shape 3 x 3")
    if point_cloud.shape[1] != 3:
        raise TypeError("Point Cloud shoud be 3 dimensional")

    rotated_points = walignedRw @ point_cloud.T
    return rotated_points.T


def transform_camera_frustums(
    itw: np.ndarray, iRw_quaternion: np.ndarray, mean: np.ndarray, wuprightRw: np.ndarray
) -> Tuple[List[float], List[float]]:
    """Transforms the camera frustums in a similar manner as the point cloud.

    Args:
        wti: camera pose translation, list of length 3.
        wRi_quaternion: camera pose rotation, list of length 4.
        means: the mean x,y,z coordinates of point cloud, array of shape (3.).
        wuprightRw: rotation matrix to align point cloud with x,y,z axes, shape 3 x 3.

    Returns:
        final_rot_quat: list, length 4, representing the quaternion for camera frustum rotation.
        final_tran: list, length 3, representing the camera frustum translation.
    """

    tx, ty, tz = itw
    qw, qx, qy, qz = iRw_quaternion
    gtsfm_rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
    gtsfm_tran = np.array([tx, ty, tz])

    wTi = np.eye(4)
    wTi[:3, :3] = gtsfm_rot
    wTi[:3, 3] = gtsfm_tran

    wcenteredTw = np.eye(4)
    wcenteredTw[:3, 3] = -1 * mean

    walignedRw = np.eye(4)
    walignedRw[:3, :3] = wuprightRw

    final_pose = walignedRw @ wcenteredTw @ wTi
    final_rot_matrix = final_pose[:3, :3]
    final_rot_quat = R.from_matrix(final_rot_matrix).as_quat()  # [qx, qy, qz, qw]
    final_tran = final_pose[:3, 3]  # [x,y,z]

    return final_rot_quat.tolist(), final_tran.tolist()
