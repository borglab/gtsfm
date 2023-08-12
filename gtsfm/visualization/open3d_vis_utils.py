"""Utilities for rendering camera frustums, 3d point clouds, and coordinate frames using the Open3d library.

Authors: John Lambert
"""

import argparse
from typing import List, Tuple

import numpy as np
import open3d
from colour import Color
from gtsam import Cal3Bundler, Pose3

from gtsfm.common.view_frustum import ViewFrustum


def create_colored_point_cloud_open3d(point_cloud: np.ndarray, rgb: np.ndarray) -> open3d.geometry.PointCloud:
    """Render a point cloud as individual colored points, using Open3d.

    Args:
        point_cloud: Array of shape (N,3) representing 3d points.
        rgb: Uint8 array of shape (N,3) representing colors in RGB order, in the range [0,255].

    Returns:
        pcd: Open3d geometry object representing a colored 3d point cloud.
    """
    colors = rgb.astype(np.float64) / 255

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_cloud)
    pcd.colors = open3d.utility.Vector3dVector(colors)

    return pcd


def convert_colored_open3d_point_cloud_to_numpy(
    pointcloud: open3d.geometry.PointCloud,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts an Open3d point cloud object to two Numpy arrays representing x/y/z coords and per-point colors.

    Args:
        pointcloud: Open3d point cloud object.

    Returns:
        points: (N,3) float array representing 3d points.
        rgb: (N,3) uint8 array representing per-point RGB colors.
    """
    points = np.asarray(pointcloud.points)
    rgb = np.asarray(pointcloud.colors)
    # open3d stores the colors as [0,1] floats.
    rgb = (rgb * 255).astype(np.uint8)
    return points, rgb


def create_colored_spheres_open3d(
    point_cloud: np.ndarray, rgb: np.ndarray, sphere_radius: float
) -> List[open3d.geometry.TriangleMesh]:
    """Create a colored sphere mesh for every point inside the point cloud, using Open3d.

    Note: this is quite computationally expensive.

    Args:
        point_cloud: Array of shape (N,3) representing 3d points.
        rgb: Uint8 array of shape (N,3) representing colors in RGB order, in the range [0,255].
        sphere_radius: Radius of each rendered sphere.

    Returns:
        spheres: List of Open3d geometry objects, where each element (a sphere) represents a 3d point.
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
    wTi_list: List[Pose3],
    calibrations: List[Cal3Bundler],
    frustum_ray_len: float = 0.3,
    color_names: Tuple[str] = ("red", "green"),
) -> List[open3d.geometry.LineSet]:
    """Render camera frustums as collections of line segments, using Open3d.

    Frustums are colored red-to-green by image order (for ordered collections, this corresponds to trajectory order).

    Args:
        wTi_list: List of camera poses for each image
        calibrations: Calibration object for each camera

    Returns:
        line_sets: List of line segments that together parameterize all camera frustums
    """
    line_sets = []

    # get array of shape Nx3 representing RGB values in [0,1], incrementally shifting from red to green
    colormap = np.array(
        [[color_obj.rgb] for color_obj in Color(color_names[0]).range_to(Color(color_names[1]), len(wTi_list))]
    ).squeeze()

    for i, (K, wTi) in enumerate(zip(calibrations, wTi_list)):

        K = K.K()
        fx = K[0, 0]

        # Use 2*principal point as proxy measure for image height and width
        # TODO (in future PR): use the real image height and width
        px = K[0, 2]
        py = K[1, 2]
        img_w = px * 2
        img_h = py * 2
        frustum_obj = ViewFrustum(fx, img_w, img_h, frustum_ray_len=frustum_ray_len)

        if np.linalg.norm(wTi.translation()) > 100:
            continue

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


def draw_coordinate_frame(wTc: Pose3, axis_length: float = 1.0) -> List[open3d.geometry.LineSet]:
    """Draw 3 orthogonal axes representing a camera coordinate frame.

    Note: x,y,z axes correspond to red, green, blue colors.

    Args:
        wTc: Pose of a camera in the world frame.
        axis_length: Length to use for line segments (representing coordinate frame axes).

    Returns:
        line_sets: List of Open3D LineSet objects, representing 3 axes (a coordinate frame).
    """
    RED = np.array([1, 0, 0])  # x-axis
    GREEN = np.array([0, 1, 0])  # y-axis
    BLUE = np.array([0, 0, 1])  # z-axis
    colors = (RED, GREEN, BLUE)

    # Line segment on each axis will connect just 2 vertices.
    lines = [[0, 1]]

    line_sets = []
    for axis, color in zip([0, 1, 2], colors):
        # one point at optical center, other point along specified axis.
        verts_camfr = np.zeros((2, 3))
        verts_camfr[0, axis] = axis_length

        verts_worldfr = []
        for i in range(2):
            verts_worldfr.append(wTc.transformFrom(verts_camfr[i]))
        verts_worldfr = np.array(verts_worldfr)

        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(verts_worldfr),
            lines=open3d.utility.Vector2iVector(lines),
        )
        line_set.colors = open3d.utility.Vector3dVector(color.reshape(1, 3))
        line_sets.append(line_set)

    return line_sets


def draw_scene_open3d(
    point_cloud: np.ndarray,
    rgb: np.ndarray,
    wTi_list: List[Pose3],
    calibrations: List[Cal3Bundler],
    args: argparse.Namespace,
) -> None:
    """Render camera frustums and a 3d point cloud, using Open3d.

    Args:
        point_cloud: Array of shape (N,3) representing 3d points.
        rgb: uint8 array of shape (N,3) representing colors in RGB order, in the range [0,255].
        wTi_list: List of camera poses for each image.
        calibrations: Calibration object for each camera.
        args: Rendering options.
    """
    #import pdb; pdb.set_trace()
    frustums = create_all_frustums_open3d(wTi_list, calibrations, args.frustum_ray_len)
    if args.rendering_style == "point":
        pcd = create_colored_point_cloud_open3d(point_cloud, rgb)
        geometries = frustums + [pcd]
    elif args.rendering_style == "sphere":
        spheres = create_colored_spheres_open3d(point_cloud, rgb, args.sphere_radius)
        geometries = frustums + spheres

    open3d.visualization.draw_geometries(geometries)


def draw_scene_with_gt_open3d(
    point_cloud: np.ndarray,
    rgb: np.ndarray,
    wTi_list: List[Pose3],
    calibrations: List[Cal3Bundler],
    gt_wTi_list: List[Pose3],
    gt_calibrations: List[Cal3Bundler],
    args: argparse.Namespace,
) -> None:
    """Render GT camera frustums, estimated camera frustums, and a 3d point cloud, using Open3d.

    GT frustums are shown in a blue-purple colormap, whereas estimated frustums are shown in a red-green colormap.

    Args:
        point_cloud: Array of shape (N,3) representing 3d points.
        rgb: Uint8 array of shape (N,3) representing colors in RGB order, in the range [0,255].
        wTi_list: List of camera poses for each image.
        calibrations: Calibration object for each camera.
        gt_wTi_list: List of ground truth camera poses for each image.
        gt_calibrations: Ground truth calibration object for each camera.
        args: Rendering options.
    """
    frustums = create_all_frustums_open3d(wTi_list, calibrations, args.frustum_ray_len, color_names=("red", "green"))
    gt_frustums = create_all_frustums_open3d(
        wTi_list, calibrations, args.frustum_ray_len, color_names=("blue", "purple")
    )

    # spheres = create_colored_spheres_open3d(point_cloud, rgb, args.sphere_radius)
    pcd = create_colored_point_cloud_open3d(point_cloud, rgb)
    geometries = frustums + gt_frustums + [pcd]  # + spheres

    open3d.visualization.draw_geometries(geometries)
