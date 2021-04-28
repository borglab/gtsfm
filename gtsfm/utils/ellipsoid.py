"""Algorithms to fit Ellisoid geometry to 3D points and extract the rotation matrix necessary to align point cloud 
with x, y, and z axes. Used in React Three Fiber Visualization Tool.

Authors: Adi Singh
"""

import numpy as np
import numpy.linalg
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple

from gtsfm.common.gtsfm_data import GtsfmData


def transform_point_cloud_wrapper(gtsfm_data: GtsfmData) -> np.ndarray:
    """Wrapper function for all the functions in ellipsoid.py. Transforms the point cloud contained within gtsfm_Data
    to be aligned with the x, y, and z axes.

    Args:
        gtsfm_data: scene data to write to transform.

    Returns:
        The transformed, aligned point cloud of shape N x 3. Also the mean x,y,z coordinates and alignment rotation
        matrix.
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
    points_centered, means = center_point_cloud(point_cloud)
    points_filtered = remove_outlier_points(points_centered)
    wuprightRw = get_rotation_matrix(points_filtered)
    aligned_points = apply_ellipsoid_rotation(wuprightRw, points_centered)

    return aligned_points, means, wuprightRw


def center_point_cloud(point_cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centers a point cloud using mean values of x, y, and z.

    Args:
        point_cloud: N 3D points to center, of shape N x 3.

    Returns:
        The centered point cloud of shape N x 3. Also an array of shape (3.) representing the mean x,y,z coordinates of point cloud.

    Raises:
        TypeError: if points are not of shape N x 3.
    """
    if point_cloud.shape[1] != 3:
        raise TypeError("Points list should be 3D")

    means = np.mean(point_cloud, axis=0)
    points_centered = point_cloud - means
    return points_centered, means


def remove_outlier_points(point_cloud: np.ndarray) -> np.ndarray:
    """Filters out the top 5% of points with greatest distance from origin.

    Args:
        point_cloud: point cloud of shape N x 3.

    Returns:
        The filtered point cloud of shape M x 3 (M = 0.95*N).

    Raises:
        TypeError: if centered point cloud is not of shape N x 3.
    """
    if point_cloud.shape[1] != 3:
        raise TypeError("Point Cloud should be 3 dimensional")

    # find mean and standard deviation of magnitudes of points to determine the threshold magnitude to filter by
    mags = np.linalg.norm(point_cloud, axis=1)
    mean_mag = np.mean(mags)
    std_mag = np.std(mags)

    # z = 1.65 represents the 95th percentile. Assuming a normal distribution of point magnitudes.
    cutoff_mag = 1.65 * (std_mag) + mean_mag

    indices_to_keep = np.where(np.linalg.norm(point_cloud, axis=1) < cutoff_mag)
    points_filtered = point_cloud[indices_to_keep]
    return points_filtered


def get_rotation_matrix(point_cloud: np.ndarray) -> np.ndarray:
    """Applies SVD on a point cloud. The resulting V contains the rotation matrix required to align the 3 major axes of
    the point cloud's ellipsoid with the x, y, z coordinate axes.

    Args:
        point_cloud: point cloud of shape N x 3.

    Returns:
        The rotation matrix, shape 3 x 3, required to align points with the x, y, and z axes.

    Raises:
        TypeError: if point cloud is not of shape N x 3.
    """
    if point_cloud.shape[1] != 3:
        raise TypeError("Point Cloud should be 3 dimensional")

    U, S, Vt = np.linalg.svd(point_cloud, full_matrices=True)
    wuprightRw = np.linalg.inv(Vt.T)
    return wuprightRw


def apply_ellipsoid_rotation(wuprightRw: np.ndarray, point_cloud: np.ndarray) -> np.ndarray:
    """Applies a rotation on the centered, filtered point cloud.

    Args:
        wuprightRw: rotation matrix, shape 3 x 3.
        point_cloud: point cloud, shape N x 3.

    Returns:
        A transformed, aligned point cloud of shape N x 3.

    Raises:
        TypeError: if rot matrix isn't 3x3 or centered_pc is not of shape N x 3.
    """
    if wuprightRw.shape[0] != 3 or wuprightRw.shape[1] != 3:
        raise TypeError("Rotation Matrix should be of shape 3 x 3")
    if point_cloud.shape[1] != 3:
        raise TypeError("Point Cloud shoud be 3 dimensional")

    # First apply the rotation matrix to the point cloud. Then swap x and z axes for visual alignment.
    swap_x_z = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    rotated_points = swap_x_z @ wuprightRw @ point_cloud.T

    return rotated_points.T


def transform_camera_frustums(
    wti: np.ndarray, wRi_quaternion: np.ndarray, means: np.ndarray, wuprightRw: np.ndarray
) -> Tuple[List[float], List[float]]:
    """Transforms the camera frustums in a similar manner as the point cloud.

    Args:
        wti: camera pose translation, list of length 3.
        wRi_quaternion: camera pose rotation, list of length 4.
        means: the mean x,y,z coordinates of point cloud, array of shape (3.).
        wuprightRw: rotation matrix to align point cloud with x,y,z axes, shape 3 x 3.

    Returns:
        Two lists. One list, of length 4, representing the quaternion for camera frustum rotation. The second list,
        of length 3, representing the camera frustum translation.
    """

    tx, ty, tz = wti
    qw, qx, qy, qz = wRi_quaternion
    gtsfm_rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
    gtsfm_tran = np.array([tx, ty, tz])

    gtsfm_pose = np.eye(4)
    gtsfm_pose[:3, :3] = gtsfm_rot
    gtsfm_pose[:3, 3] = gtsfm_tran

    wcenterTw = np.eye(4)
    wcenterTw[:3, 3] = -1 * means

    walignRw = np.eye(4)
    walignRw[:3, :3] = wuprightRw

    w_xzswapRw = np.array([[0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

    final_pose = w_xzswapRw @ walignRw @ wcenterTw @ gtsfm_pose
    final_rot_matrix = final_pose[:3, :3]
    final_rot_quat = R.from_matrix(final_rot_matrix).as_quat()  # [qx, qy, qz, qw]
    final_tran = final_pose[:3, 3]  # [x,y,z]

    return final_rot_quat.tolist(), final_tran.tolist()
