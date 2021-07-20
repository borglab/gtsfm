"""
Script to render a GTSFM scene using either Open3d or Mayavi mlab.

Authors: John Lambert
"""

import argparse
import os
from pathlib import Path

import numpy as np
from gtsam import Rot3, Pose3

import gtsfm.utils.io as io_utils

from visualization.mayavi_vis_utils import draw_scene_mayavi
from visualization.open3d_vis_utils import draw_scene_open3d


REPO_ROOT = Path(__file__).parent.parent.resolve()


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


def view_scene(args: argparse.Namespace) -> None:
    """Read GTSFM output from .txt files and render the scene to the GUI.

    We also zero-center the point cloud, and transform camera poses to a new
    world frame, where the point cloud is zero-centered.

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
    # expression below is equivalent to applying zcwTw.transformFrom() to each world point
    point_cloud -= mean_pt

    is_nearby = np.linalg.norm(point_cloud, axis=1) < args.max_range
    point_cloud = point_cloud[is_nearby]
    rgb = rgb[is_nearby]

    for i in range(len(wTi_list)):
        wTi_list[i] = zcwTw.compose(wTi_list[i])

    if args.rendering_library == "open3d":
        draw_scene_open3d(point_cloud, rgb, wTi_list, calibrations, args)
    elif args.rendering_library == "mayavi":
        draw_scene_mayavi(point_cloud, rgb, wTi_list, calibrations, args)
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
        "--output_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "ba_output"),
        help="Path to a directory containing GTSFM output. "
        "This directory should contain 3 files: cameras.txt, images.txt, and points3D.txt",
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
        default=0.1,
        help="if points are rendered as spheres, then spheres are rendered with this radius.",
    )

    args = parser.parse_args()
    if args.point_rendering_mode == "point" and args.rendering_library == "mayavi":
        raise RuntimeError("Mayavi only supports rendering points as spheres.")

    view_scene(args)
