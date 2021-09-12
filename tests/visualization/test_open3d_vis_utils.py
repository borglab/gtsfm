"""Unit tests for Open3d visualizaation code.

Author: John Lambert
"""

import numpy as np
import open3d

import visualization.open3d_vis_utils as open3d_vis_utils


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
