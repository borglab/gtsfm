"""Algorithms to center and align 3D points and camera frustums to the x, y, and z axes using SVD. 

Process is similar to Principal Component Analysis. Used in React Three Fiber Visualization Tool.

Authors: Adi Singh
"""
from typing import Tuple

import numpy as np
from gtsam import Pose3, Rot3

from gtsfm.common.gtsfm_data import GtsfmData

# percentile threshold to classify points as outlier based on magnitude
OUTLIER_DISTANCE_PERCENTILE = 95


def get_ortho_axis_alignment_transform(gtsfm_data: GtsfmData) -> Pose3:
    """Wrapper function for all the functions in ellipsoid.py. Obtains the Pose3 transformation required to align
    the GtsfmData to the x,y,z axes.

    Args:
        gtsfm_data: Scene data to write to transform.

    Returns:
        The final transformation required to align point cloud and frustums.
    """
    # Iterate through each track to gather a list of 3D points forming the point cloud.
    num_pts = gtsfm_data.number_tracks()
    point_cloud = [gtsfm_data.get_track(j).point3() for j in range(num_pts)]
    point_cloud = np.array(point_cloud)  # point_cloud has shape Nx3

    # Filter outlier points, Center point cloud, and obtain alignment rotation.
    points_filtered, inlier_mask = remove_outlier_points(point_cloud)
    points_centered = center_point_cloud(points_filtered)
    wuprightRw = get_alignment_rotation_matrix_from_svd(points_centered)

    # Calculate translation vector based off rotated point cloud (excluding outliers).
    point_cloud_rotated = point_cloud @ wuprightRw.T
    rotated_mean = np.mean(point_cloud_rotated[inlier_mask], axis=0)

    # Obtain the Pose3 object needed to align camera frustums.
    walignedTw = Pose3(Rot3(wuprightRw), -1 * rotated_mean)

    return walignedTw


def center_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    """Centers a point cloud using mean values of x, y, and z.

    Args:
        point_cloud: Array of shape (N,3) representing the original point cloud.

    Returns:
        points_centered: Array of shape (N,3) representing the centered point cloud

    Raises:
        TypeError: If point_cloud is not of shape (N,3).
    """
    if point_cloud.shape[1] != 3:
        raise TypeError("Points list should be 3D")

    mean = np.mean(point_cloud, axis=0)
    points_centered = point_cloud - mean
    return points_centered


def remove_outlier_points(point_cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Removes the top 5% of points with greatest distance from origin.

    Args:
        point_cloud: Point cloud of shape (N, 3).

    Returns:
        points_filtered: Filtered point cloud of shape (M, 3), with (M = 0.95*N)
        inlier_mask: Boolean array, shape (N,), representing which points in point cloud are inliers.

    Raises:
        TypeError: If centered point cloud is not of shape (N,3).
    """
    if point_cloud.ndim != 2:
        raise ValueError(f"Point cloud must have shape (N,3) but received {point_cloud.shape}")

    if point_cloud.shape[1] != 3:
        raise TypeError("Point cloud should be 3 dimensional")

    mags = np.linalg.norm(point_cloud, axis=1)
    cutoff_mag = np.percentile(mags, OUTLIER_DISTANCE_PERCENTILE)
    inlier_mask = mags < cutoff_mag
    points_filtered = point_cloud[inlier_mask]
    return points_filtered, inlier_mask


def get_alignment_rotation_matrix_from_svd(point_cloud: np.ndarray) -> np.ndarray:
    """Applies SVD to fit an ellipsoid to the point cloud. The resulting V contains the rotation matrix required to
    align the 3 principal axes of the ellipsoid with the x, y, z coordinate axes.

    Args:
        point_cloud: Point cloud of shape (N,3).

    Returns:
        The rotation matrix, shape (3,3), required to align points with the x, y, and z axes.

    Raises:
        TypeError: If point cloud is not of shape (N,3).
    """
    if point_cloud.shape[1] != 3:
        raise TypeError("Point Cloud should be 3 dimensional")

    # Obtain right singular vectors to determine rotation matrix of point cloud.
    V, _ = get_right_singular_vectors(point_cloud)
    Vt = V.T

    # If det(Vt) = -1, then Vt is a reflection matrix and not a valid SO(3) transformation. Thus, we must estimate the
    # closest rotation matrix to the reflection.
    if not np.isclose(np.linalg.det(Vt), 1):
        wuprightRw = Rot3.ClosestTo(Vt).matrix()  # changes Vt's eigenvalue from -1 to +1 to convert to rotation matrix
    else:
        wuprightRw = Vt

    return wuprightRw


def get_right_singular_vectors(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts the right singular eigenvectors from the point cloud. Some of the eigenvectors could be randomly
    multiplied by -1. Despite this, the eigenvectors will still remain valid.

    Ref: https://stackoverflow.com/questions/18152052/matlab-eig-returns-inverted-signs-sometimes

    Args:
        A: point cloud of shape (N,3)

    Returns:
        The right singular vectors of the point cloud, shape (3,3).
        The singular values of the point cloud (sorted in descending order), with shape (3,)

    Raises:
        TypeError: If point cloud is not of shape (N,3).
    """
    N, D = A.shape
    if D != 3:
        raise TypeError("Point cloud should be 3 dimensional.")

    # Eigenvectors of A^T*A are singular vectors of A
    # We apply Bessel's correction when estimating the covariance matrix.
    # See https://en.wikipedia.org/wiki/Principal_component_analysis#Computing_PCA_using_the_covariance_method
    eigvals, eigvecs = np.linalg.eig(A.T @ A / (N - 1))

    # Sort eigenvectors such that they correspond to eigenvalues sorted in descending order.
    sort_idxs = np.argsort(-eigvals)

    return eigvecs[:, sort_idxs], np.sqrt(eigvals[sort_idxs])
