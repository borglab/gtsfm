"""Unit tests for Open3d visualization code.

Author: John Lambert
"""

import numpy as np
import open3d

import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils


def test_round_trip_open3d_numpy_pointcloud() -> None:
    """Round trip test to ensure numpy->Open3d->numpy conversion yields identity operation."""

    point_cloud = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    rgb = np.array([[255, 0, 0], [101, 1, 0], [3, 6, 255], [0, 1, 2]], dtype=np.uint8)

    pcd = open3d_vis_utils.create_colored_point_cloud_open3d(point_cloud=np.copy(point_cloud), rgb=np.copy(rgb))
    points_roundtrip, rgb_roundtrip = open3d_vis_utils.convert_colored_open3d_point_cloud_to_numpy(pointcloud=pcd)

    assert np.allclose(points_roundtrip, point_cloud)
    assert points_roundtrip.dtype == point_cloud.dtype

    assert np.allclose(rgb_roundtrip, rgb)
    assert rgb_roundtrip.dtype == rgb.dtype


def test_create_colored_spheres_open3d() -> None:
    """Try creating spheres centered at randomly generated 3d locations (disclaimer: slow)."""
    np.random.seed(0)
    num_points = 1000
    point_cloud = np.random.randn(num_points, 3)
    # high bound is exclusive
    rgb = np.random.randint(low=0, high=256, size=(num_points, 3))
    sphere_radius = 0.1

    spheres = open3d_vis_utils.create_colored_spheres_open3d(point_cloud, rgb, sphere_radius)

    assert isinstance(spheres, list)
    assert all([isinstance(sphere, open3d.geometry.TriangleMesh) for sphere in spheres])


def test_create_colored_point_cloud_open3d() -> None:
    """Try creating points at randomly generated 3d locations (should be fast)."""
    np.random.seed(0)
    num_points = 10000
    point_cloud = np.random.randn(num_points, 3)
    # high bound is exclusive
    rgb = np.random.randint(low=0, high=256, size=(num_points, 3))

    pcd = open3d_vis_utils.create_colored_point_cloud_open3d(point_cloud, rgb)
    assert isinstance(pcd, open3d.geometry.PointCloud)
