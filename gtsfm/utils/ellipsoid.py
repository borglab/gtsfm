"""Algorithms to fit Ellisoid geometry to 3D points and extract the rotation matrix necessary to align point cloud 
with x, y, and z axes. Used in React Three Fiber Visualization Tool.

Authors: Adi Singh
"""

import numpy as np
from numpy.linalg import eig, inv, norm, svd


def center_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    """Centers a point cloud using mean values of x, y, and z.

    Args:
        pointCloud: array representing the point cloud, of shape N x 3.

    Returns:
        points_centered: centered point cloud of shape Nx3.
        means: the means of the x,y,z coordinates, array of length 3.

    Raises:
        TypeError: if points list is not 3 dimensional.
    """
    if point_cloud.shape[1] != 3:
        raise TypeError("Points list should be 3D")

    means = np.mean(point_cloud, axis=0)
    points_centered = point_cloud - means
    return points_centered, means


def filter_outlier_points(centered_pc: np.ndarray) -> np.ndarray:
    """Filters the top 5% of points with greatest distance from origin.

    Args:
        centered_pc: centered point cloud of shape N x 3.

    Returns:
        The filtered point cloud of shape M x 3 (M = 0.95*N).

    Raises:
        TypeError: if centered point cloud is not 3 dimensional.
    """
    if centered_pc.shape[1] != 3:
        raise TypeError("Point Cloud should be 3 dimensional")

    # find mean and standard deviation of magnitudes of points to determine the threshold magnitude to filter by
    mags = norm(centered_pc, axis=1)
    mean_mag = np.mean(mags)
    std_mag = np.std(mags)

    # z = 1.65 represents the 95th percentile. Assuming a normal distribution of point magnitudes.
    cutoff_mag = 1.65 * (std_mag) + mean_mag

    indices_to_keep = np.where(norm(centered_pc, axis=1) < cutoff_mag)
    points_filtered = centered_pc[indices_to_keep]
    return points_filtered


def get_rotation_matrix(filtered_pc: np.ndarray) -> np.ndarray:
    """Applies SVD on a point cloud to extract the rotation matrix to align points with axes.

    Args:
        filtered_pc: the filtered point cloud of shape M x 3.

    Returns:
        The rotation matrix, shape 3 x 3, required to align points with the x, y, and z axes.

    Raises:
        TypeError: if centered point cloud is not 3 dimensional.
    """
    if filtered_pc.shape[1] != 3:
        raise TypeError("Point Cloud should be 3 dimensional")

    u, s, v = svd(filtered_pc, full_matrices=True)
    v = v.T
    rot = inv(v)
    return rot


def apply_ellipsoid_rotation(rot: np.ndarray, centered_pc: np.ndarray) -> np.ndarray:
    """Applies a rotation on the centered, filtered point cloud.

    Args:
        rot: rotation matrix, shape 3 x 3.
        centered_pc: centered point cloud, shape N x 3.

    Returns:
        A transformed, aligned point cloud of shape N x 3.

    Raises:
        TypeError: if rot matrix isn't 3x3 or centered_pc is not of dimension 3
    """
    if rot.shape[0] != 3 or rot.shape[1] != 3:
        raise TypeError("Rotation Matrix should be of shape 3 x 3")
    if centered_pc.shape[1] != 3:
        raise TypeError("Point Cloud shoud be 3 dimensional")

    # First apply the rotation matrix to the point cloud. Then swap x and z axes for visual alignment.
    swap_x_z = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    rotated_points = swap_x_z @ rot @ centered_pc.T

    return rotated_points.T
