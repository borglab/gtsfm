"""
Script to render a GTSFM scene using either Open3d or Mayavi mlab.

Author: John Lambert
"""

import argparse
import os
from pathlib import Path
from typing import List

import mayavi
import numpy as np
import open3d
from colour import Color
from gtsam import Cal3Bundler, Rot3, Pose3
from mayavi import mlab

import gtsfm.utils.io as io_utils
from gtsfm.common.view_frustum import ViewFrustum

REPO_ROOT = Path(__file__).parent.resolve()


def draw_point_cloud_mayavi(
    args: argparse.Namespace, fig: mayavi.core.scene.Scene, point_cloud: np.ndarray, rgb: np.ndarray
) -> None:
    """Render a point cloud as a collection of spheres, using Mayavi.

    Args:
        args: rendering options.
        fig: Mayavi figure object.
        point_cloud: array of shape (N,3) representing 3d points.
        rgb: uint8 array of shape (N,3) representing colors in RGB order, in the range [0,255]
    """
    n = point_cloud.shape[0]
    x, y, z = point_cloud.T
    alpha = np.ones((n, 1)).astype(np.uint8) * 255  # no transparency
    rgba = np.hstack([rgb, alpha]).astype(np.uint8)

    pts = mlab.pipeline.scalar_scatter(x, y, z)  # plot the points
    pts.add_attribute(rgba, "colors")  # assign the colors to each point
    pts.data.point_data.set_active_scalars("colors")
    g = mlab.pipeline.glyph(pts)
    g.glyph.glyph.scale_factor = args.sphere_radius  # set scaling for all the points
    g.glyph.scale_mode = "data_scaling_off"  # make all the points same size


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
    args: argparse.Namespace, point_cloud: np.ndarray, rgb: np.ndarray
) -> List[open3d.geometry.TriangleMesh]:
    """Create a colored sphere mesh for every point inside the point cloud, using Open3d.

    Note: this is quite computationally expensive.

    Args:
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
        mesh = open3d.geometry.TriangleMesh.create_sphere(radius=args.sphere_radius, resolution=20)
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


def draw_cameras_mayavi(
    zcwTw: Pose3, fig: mayavi.core.scene.Scene, calibrations: List[Cal3Bundler], wTi_list: List[Pose3]
) -> None:
    """Render camera frustums as collections of line segments, using Mayavi mlab.

    Args:
        zcwTw: transforms world points to a new world frame where the point cloud is zero-centered
        fig: Mayavi mlab figure object.
        calibrations: calibration object for each camera
        wTi_list: list of camera poses for each image
    """
    colormap = np.array(
        [[color_obj.rgb] for color_obj in Color("red").range_to(Color("green"), len(wTi_list))]
    ).squeeze()

    for i, (K, wTi) in enumerate(zip(calibrations, wTi_list)):
        wTi = zcwTw.compose(wTi)

        color = tuple(colormap[i].tolist())

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
        for edge_worldfr in edges_worldfr:

            # start and end vertices
            vs = edge_worldfr[0]
            ve = edge_worldfr[1]

            # TODO: consider adding line_width
            mlab.plot3d(  # type: ignore
                [vs[0], ve[0]],
                [vs[1], ve[1]],
                [vs[2], ve[2]],
                color=color,
                tube_radius=None,
                figure=fig,
            )


def compute_point_cloud_center_robust(point_cloud: np.ndarray) -> np.ndarray:
    """Robustly estimate the point cloud center.

    Args:
        point_cloud: array of shape (N,3) representing 3d points.

    Returns:
        mean_pt: coordinates of central point, ignoring outliers.
    """
    ranges = np.linalg.norm(point_cloud, axis=1)
    outlier_thresh = np.percentile(ranges, 75)
    mean_pt = point_cloud[ranges < outlier_thresh].mean(axis=0)
    return mean_pt


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
        spheres = create_colored_spheres_open3d(args, point_cloud, rgb)
        geometries = frustums + spheres

    open3d.visualization.draw_geometries(geometries)


def draw_scene_mayavi(
    args: argparse.Namespace,
    point_cloud: np.ndarray,
    rgb: np.ndarray,
    calibrations: List[Cal3Bundler],
    wTi_list: List[Pose3],
    zcwTw: Pose3,
) -> None:
    """Render camera frustums and a 3d point cloud against a white background, using Mayavi.

    Args:
        args: rendering options.
        point_cloud
        rgb: uint8 array of shape (N,3) representing colors in RGB order, in the range [0,255].
        calibrations: calibration object for each camera
        wTi_list: list of camera poses for each image
        zcwTw: transforms world points to a new world frame where the point cloud is zero-centered
    """
    bgcolor = (1, 1, 1)
    fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))  # type: ignore
    draw_cameras_mayavi(zcwTw, fig, calibrations, wTi_list)
    draw_point_cloud_mayavi(args, fig, point_cloud, rgb)
    mlab.show()


def view_scene(args: argparse.Namespace) -> None:
    """Read GTSFM output from .txt files and render the scene to the GUI.

    Args:
        args: rendering options.
    """
    points_fpath = f"{args.output_dir}/points3D.txt"
    images_fpath = f"{args.output_dir}/images.txt"
    cameras_fpath = f"{args.output_dir}/cameras.txt"

    wTi_list, img_fnames = io_utils.read_images_txt(images_fpath)
    calibrations = io_utils.read_cameras_txt(cameras_fpath)

    if len(calibrations) == 1:
        calibrations = calibrations * len(img_fnames)

    point_cloud, rgb = io_utils.read_points_txt(points_fpath)

    mean_pt = compute_point_cloud_center_robust(point_cloud)

    # Zero-center the point cloud (about estimated center)
    zcwTw = Pose3(Rot3(np.eye(3)), -mean_pt)
    n = point_cloud.shape[0]
    for j in range(n):
        point_cloud[j] = zcwTw.transformFrom(point_cloud[j])

    is_nearby = np.linalg.norm(point_cloud, axis=1) < args.max_range
    point_cloud = point_cloud[is_nearby]
    rgb = rgb[is_nearby]

    if args.rendering_library == "open3d":
        draw_scene_open3d(args, point_cloud, rgb, calibrations, wTi_list, zcwTw)
    elif args.rendering_library == "mayavi":
        draw_scene_mayavi(args, point_cloud, rgb, calibrations, wTi_list, zcwTw)
    else:
        raise RuntimeError("Unsupported rendering library")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize GTSFM result with Mayavi or Open3d.")
    parser.add_argument(
        "--rendering_library",
        type=str,
        default="mayavi",
        choices=["mayavi", "open3d"],
        help="3d rendering library to use.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join(REPO_ROOT, "results", "ba_output"), help="Path to GTSFM output"
    )
    parser.add_argument(
        "--point_rendering_mode",
        type=str,
        default="sphere",
        choices=["point", "sphere"],
        help="Render each 3d point as a `point` (optimized in Open3d) or `sphere` (optimized in Mayavi).",
    )
    parser.add_argument(
        "--max_range",
        type=float,
        default=20,
        help="maximum range of points (from estimated point cloud center) to render.",
    )
    parser.add_argument(
        "--sphere_radius",
        type=float,
        default=0.001,
        help="if points are rendered as spheres, then spheres are rendered with this radius.",
    )

    args = parser.parse_args()
    if args.point_rendering_mode == "point" and args.rendering_library == "mayavi":
        raise RuntimeError("Mayavi only supports rendering points as spheres.")

    view_scene(args)
