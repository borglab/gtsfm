"""
Utilities for rendering camera frustums and 3d point clouds using Open3d.

Author: John Lambert
"""

import argparse
from typing import List

import numpy as np
import open3d
from colour import Color
from gtsam import Cal3Bundler, Pose3

from gtsfm.common.view_frustum import ViewFrustum


def create_colored_point_cloud_open3d(point_cloud: np.ndarray, rgb: np.ndarray) -> open3d.geometry.PointCloud:
    """Render a point cloud as individual colored points, using Open3d.

    Args:
        point_cloud: array of shape (N,3) representing 3d points.
        rgb: uint8 array of shape (N,3) representing colors in RGB order, in the range [0,255]

    Returns:
        pcd: Open3d geometry object representing a colored 3d point cloud.
    """
    colors = rgb.astype(np.float64) / 255

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_cloud)
    pcd.colors = open3d.utility.Vector3dVector(colors)

    return pcd


def create_colored_spheres_open3d(
    sphere_radius: float, point_cloud: np.ndarray, rgb: np.ndarray
) -> List[open3d.geometry.TriangleMesh]:
    """Create a colored sphere mesh for every point inside the point cloud, using Open3d.

    Note: this is quite computationally expensive.

    Args:
        sphere_radius: radius of each rendered sphere.
        point_cloud: array of shape (N,3) representing 3d points.
        rgb: uint8 array of shape (N,3) representing colors in RGB order, in the range [0,255]

    Returns:
        spheres: list of Open3d geometry objects, where each element (a sphere) represents a 3d point.
    """
    colors = rgb.astype(np.float64) / 255

    spheres = []
    n = point_cloud.shape[0]

    for j in range(n):
        wTj = np.eye(4)
        wTj[:3, 3] = point_cloud[j]
        mesh = open3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=20)
        mesh.transform(wTj)
        mesh.paint_uniform_color(colors[j])

        spheres.append(mesh)

    return spheres


def create_all_frustums_open3d(
    zcwTw: Pose3, calibrations: List[Cal3Bundler], wTi_list: List[Pose3]
) -> List[open3d.geometry.LineSet]:
    """Render camera frustums as collections of line segments, using Open3d.

    Args:
        zcwTw: transforms world points to a new world frame where the point cloud is zero-centered
        calibrations: calibration object for each camera
        wTi_list: list of camera poses for each image

    Returns:
        line_sets: list of line segments that together parameterize all camera frustums
    """
    line_sets = []

    # get array of shape Nx3 representing RGB values in [0,1], incrementally shifting from red to green
    colormap = np.array(
        [[color_obj.rgb] for color_obj in Color("red").range_to(Color("green"), len(wTi_list))]
    ).squeeze()

    for i, (K, wTi) in enumerate(zip(calibrations, wTi_list)):

        # get camera pose, after zero-centering
        wTi = zcwTw.compose(wTi)

        K = K.K()
        fx = K[0, 0]

        # Use 2*principal point as proxy measure for image height and width
        # TODO (in future PR): use the real image height and width
        px = K[0, 2]
        py = K[1, 2]
        img_w = px * 2
        img_h = py * 2
        frustum_obj = ViewFrustum(fx, img_w, img_h)

        edges_worldfr = frustum_obj.get_mesh_edges_worldframe(wTi)
        for verts_worldfr in edges_worldfr:

            lines = [[0, 1]]
            # color is in range [0,1]
            color = tuple(colormap[i].tolist())
            colors = [color for i in range(len(lines))]

            line_set = open3d.geometry.LineSet(
                points=open3d.utility.Vector3dVector(verts_worldfr),
                lines=open3d.utility.Vector2iVector(lines),
            )
            line_set.colors = open3d.utility.Vector3dVector(colors)
            line_sets.append(line_set)

    return line_sets


def draw_scene_open3d(
    args: argparse.Namespace,
    point_cloud: np.ndarray,
    rgb: np.ndarray,
    calibrations: List[Cal3Bundler],
    wTi_list: List[Pose3],
    zcwTw: Pose3,
) -> None:
    """Render camera frustums and a 3d point cloud, using Open3d.

    Args:
        args: rendering options.
        point_cloud: array of shape (N,3) representing 3d points.
        rgb: uint8 array of shape (N,3) representing colors in RGB order, in the range [0,255].
        calibrations: calibration object for each camera
        wTi_list: list of camera poses for each image
        zcwTw: transforms world points to a new world frame where the point cloud is zero-centered
    """
    frustums = create_all_frustums_open3d(zcwTw, calibrations, wTi_list)
    if args.point_rendering_mode == "point":
        pcd = create_colored_point_cloud_open3d(point_cloud, rgb)
        geometries = frustums + [pcd]
    elif args.point_rendering_mode == "sphere":
        spheres = create_colored_spheres_open3d(args.sphere_radius, point_cloud, rgb)
        geometries = frustums + spheres

    open3d.visualization.draw_geometries(geometries)
