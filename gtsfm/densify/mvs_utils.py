"""Utilities for the multi-view stereo (MVS) stage of the back-end.

Authors: Ren Liu, Ayush Baid
"""

import math
from typing import Optional, Tuple

import numpy as np
from gtsam import PinholeCameraCal3Bundler, Unit3
from scipy.spatial import KDTree

import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils
from gtsfm.common.metrics_sink import MetricsSink
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.utils import ellipsoid as ellipsoid_utils
from gtsfm.utils import geometry_comparisons as geometry_utils

# epsilon, added to denominator to prevent division by zero.
EPS = 1e-12


def calculate_triangulation_angle_in_degrees(
    camera_1: PinholeCameraCal3Bundler, camera_2: PinholeCameraCal3Bundler, point_3d: np.ndarray
) -> float:
    """Calculates the angle formed at the 3D point by the rays backprojected from 2 cameras.
    In the setup with X (point_3d) and two cameras C1 and C2, the triangulation angle is the angle between rays C1-X
    and C2-X, i.e. the angle subtended at the 3d point.
        X
       / \
      /   \
     /     \
    C1      C2
    References:
    - https://github.com/colmap/colmap/blob/dev/src/base/triangulation.cc#L122
    
    Args:
        camera_1: the first camera.
        camera_2: the second camera.
        point_3d: the 3d point which is imaged by the two camera centers, and where the angle between the light rays 
                  associated with the measurements are computed.
    Returns:
        the angle formed at the 3d point, in degrees.
    """
    camera_center_1: np.ndarray = camera_1.pose().translation()
    camera_center_2: np.ndarray = camera_2.pose().translation()

    # compute the two rays
    ray_1 = point_3d - camera_center_1
    ray_2 = point_3d - camera_center_2

    return geometry_utils.compute_relative_unit_translation_angle(Unit3(ray_1), Unit3(ray_2))


def calculate_triangulation_angles_in_degrees(
    camera_1: PinholeCameraCal3Bundler, camera_2: PinholeCameraCal3Bundler, points_3d: np.ndarray
) -> np.ndarray:
    """Vectorized. calculation of the angles formed at 3D points by the rays backprojected from 2 cameras.
    In the setup with X (point_3d) and two cameras C1 and C2, the triangulation angle is the angle between rays C1-X
    and C2-X, i.e. the angle subtended at the 3d point.
        X
       / \
      /   \
     /     \
    C1      C2
    References:
    - https://github.com/colmap/colmap/blob/dev/src/base/triangulation.cc#L122
    
    Args:
        camera_1: the first camera.
        camera_2: the second camera.
        points_3d: (N,3) 3d points which are imaged by the two camera centers, and where the angle between the
                  light rays associated with the measurements are computed.
    Returns:
        the angles formed at the 3d points, in degrees.

    https://github.com/colmap/colmap/blob/dev/src/base/triangulation.cc#L147
    """
    camera_center_1: np.ndarray = camera_1.pose().translation()
    camera_center_2: np.ndarray = camera_2.pose().translation()

    N = points_3d.shape[0]
    # ensure broadcasting is in the correct direction
    rays1 = points_3d - camera_center_1.reshape(1, 3)
    rays2 = points_3d - camera_center_2.reshape(1, 3)

    # normalize rays to unit length
    rays1 /= np.linalg.norm(rays1, axis=1).reshape(N, 1)
    rays2 /= np.linalg.norm(rays2, axis=1).reshape(N, 1)

    dot_products = np.multiply(rays1, rays2).sum(axis=1)
    dot_products = np.clip(dot_products, -1, 1)
    angles_rad = np.arccos(dot_products)
    angles_deg = np.rad2deg(angles_rad)
    return angles_deg


def piecewise_gaussian(theta: float, theta_0: float = 5, sigma_1: float = 1, sigma_2: float = 10) -> float:
    """A Gaussian function that favors a certain baseline angle (theta_0).

    The Gaussian function is divided into two pieces with different standard deviations:
    - If the input baseline angle (theta) is no larger than theta_0, the standard deviation will be sigma_1;
    - If theta is larger than theta_0, the standard deviation will be sigma_2.

    More details can be found in "View Selection" paragraphs in Yao's paper https://arxiv.org/abs/1804.02505.

    Args:
        theta: the input baseline angle.
        theta_0: defaults to 5, the expected baseline angle of the function.
        sigma_1: defaults to 1, the standard deviation of the function when the angle is no larger than theta_0.
        sigma_2: defaults to 10, the standard deviation of the function when the angle is larger than theta_0.

    Returns:
        the result of the Gaussian function, in range (0, 1]
    """

    # Decide the standard deviation according to theta and theta_0
    # if theta is no larger than theta_0, the standard deviation is sigma_1
    if theta <= theta_0:
        sigma = sigma_1
    # if theta is larger than theta_0, the standard deviation is sigma_2
    else:
        sigma = sigma_2

    return math.exp(-((theta - theta_0) ** 2) / (2 * sigma**2))


def cart_to_homogenous(
    non_homogenous_coordinates: np.ndarray,
) -> np.ndarray:
    """Convert cartesian coordinates to homogenous system (by appending a row of ones).
    Args:
        non_homogenous_coordinates: d-dim non-homogenous coordinates, of shape dxN.

    Returns:
        (d+1)-dim homogenous coordinates, of shape (d+1)xN.

    Raises:
        TypeError if input non_homogenous_coordinates is not 2 dimensional.
    """

    if len(non_homogenous_coordinates.shape) != 2:
        raise TypeError("Input non-homogenous coordinates should be 2 dimensional")

    n = non_homogenous_coordinates.shape[1]

    return np.vstack([non_homogenous_coordinates, np.ones((1, n))])


def estimate_voxel_scales(points: np.ndarray) -> np.ndarray:
    """Estimate the voxel scales along 3 orthogonal axes through computing semi-axis lengths of a centered point cloud
    by eigen-decomposition, see Ellipsoid from point cloud: https://forge.epn-campus.eu/svn/vtas/vTIU/doc/ellipsoide.pdf

    Args:
        points: array of shape (N,3)

    Returns:
        voxel scales along 3 orthogonal axes in the descent order
    """
    # center the point cloud
    centered_points = ellipsoid_utils.center_point_cloud(points)

    # get semi-axis lengths in all axes of the centered point cloud
    _, singular_values = ellipsoid_utils.get_right_singular_vectors(centered_points)

    return singular_values


def estimate_minimum_voxel_size(points: np.ndarray, scale: float = 0.02) -> float:
    """Estimate the minimum voxel size for point cloud simplification by downsampling
        1. compute the minimum semi-axis length of a centered point cloud by eigen-decomposition
        2. scale it to obtain the minimum voxel size for point cloud simplification by downsampling

    Args:
        points: array of shape (N,3)
        scale: expected scale from the minimum semi-axis length to the output voxel size.
            a larger scale results in a larger voxel size, which means a more compressed scene. Defaults to 0.02.

    Returns:
        the minimum voxel size for point cloud simplification by downsampling
    """
    N = points.shape[0]

    # if the number of points is less than 2, then return 0
    if N < 2:
        return 0

    # get semi-axis lengths along 3 orthogonal axes in descent order
    voxel_scales = estimate_voxel_scales(points=points)

    # set the minimum voxel size as the scaled minimum semi-axis length
    return voxel_scales[-1] * scale


def downsample_point_cloud(
    points: np.ndarray, rgb: np.ndarray, voxel_size: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """Use voxel downsampling to create a uniformly downsampled point cloud from an input point cloud.

    The algorithm uses a regular voxel grid and operates in two steps
        1. Points are bucketed into voxels.
        2. Each occupied voxel generates exactly one point by averaging all points inside.
    See http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Voxel-downsampling

    Args:
        points: array of shape (N,3)
        rgb: array of shape (N,3)
        voxel_size: size of voxel.

    Returns:
        points_downsampled: array of shape (M,3) where M <= N
        rgb_downsampled: array of shape (M,3) where M <= N
    """

    # if voxel_size is invalid, do not downsample the point cloud
    if voxel_size <= 0:
        return points, rgb

    pcd = open3d_vis_utils.create_colored_point_cloud_open3d(point_cloud=points, rgb=rgb)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    points_downsampled, rgb_downsampled = open3d_vis_utils.convert_colored_open3d_point_cloud_to_numpy(pcd)
    return points_downsampled, rgb_downsampled


def compute_downsampling_psnr(original_point_cloud: np.ndarray, downsampled_point_cloud: np.ndarray) -> float:
    """Compute PSNR between original point cloud and downsampled point cloud
    A larger PSNR shows smaller distances between:
        1. the nearest neighbors of a point in the original point cloud found in the downsampled point cloud
        1. the nearest neighbors of a point in the downsampled point cloud found in the original point cloud

    Ref: Schnabel, R., & Klein, R. (2006, July). Octree-based Point-Cloud Compression. In PBG@SIGGRAPH (pp. 111-120).
        https://diglib.eg.org/xmlui/bitstream/handle/10.2312/SPBG.SPBG06.111-120/111-120.pdf?sequence=1

    Args:
        original_point_cloud: original dense point cloud before downsampling, in shape of (N, 3)
        downsampled_point_cloud: dense point cloud after downsampling, in shape of (N', 3), where N' <= N

    Returns:
        float representing PSNR between original point cloud and downsampled point cloud
    """
    # calculate the bounding box diagonal as estimated voxel scale
    #   this diagonal is estimated as the diagonal of the ellipsoid's circumscribed rectangular parallelepiped
    est_voxel_scale = 2.0 * np.linalg.norm(estimate_voxel_scales(original_point_cloud))

    original_tree = KDTree(data=original_point_cloud)
    downsampled_tree = KDTree(data=downsampled_point_cloud)

    d_downsampled_to_original, _ = original_tree.query(downsampled_point_cloud)
    d_original_to_downsampled, _ = downsampled_tree.query(original_point_cloud)

    def RMS(data):
        return np.sqrt(np.square(data).mean())

    psnr = 20.0 * np.log10(est_voxel_scale / max(RMS(d_downsampled_to_original), RMS(d_original_to_downsampled)))

    return psnr


def get_voxel_downsampling_metrics(
    min_voxel_size: float,
    original_point_cloud: np.ndarray,
    downsampled_point_cloud: np.ndarray,
    metrics_sink: Optional[MetricsSink] = None,
) -> None:
    """Collect and compute metrics for voxel downsampling
    Args:
        min_voxel_size: minimum voxel size for voxel downsampling
        original_point_cloud: original dense point cloud before downsampling
        downsampled_point_cloud: dense point cloud after downsampling

    Returns:
        GtsfmMetricsGroup: voxel down-samping metrics group
    """
    psnr = compute_downsampling_psnr(
        original_point_cloud=original_point_cloud, downsampled_point_cloud=downsampled_point_cloud
    )

    downsampling_metrics = []
    downsampling_metrics.append(GtsfmMetric(name="voxel size for downsampling", data=min_voxel_size))
    downsampling_metrics.append(
        GtsfmMetric(name="point cloud size before downsampling", data=original_point_cloud.shape[0])
    )
    downsampling_metrics.append(
        GtsfmMetric(name="point cloud size after downsampling", data=downsampled_point_cloud.shape[0])
    )
    downsampling_metrics.append(
        GtsfmMetric(
            name="compression ratio", data=original_point_cloud.shape[0] / (downsampled_point_cloud.shape[0] + EPS)
        )
    )
    downsampling_metrics.append(GtsfmMetric(name="downsampling PSNR", data=psnr))

    if metrics_sink is not None:
        metrics_sink.record(GtsfmMetricsGroup(name="voxel downsampling metrics", metrics=downsampling_metrics))
