"""Script to render a GTSFM scene using either Open3d.

Results must be stored in the COLMAP .txt or .bin file format.

Authors: John Lambert
"""

import argparse
import os
from pathlib import Path

import numpy as np
from gtsam import Pose3, Rot3

import gtsfm.utils.io as io_utils
from gtsfm.visualization.open3d_vis_utils import draw_scene_open3d

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()


def compute_point_cloud_center_robust(point_cloud: np.ndarray) -> np.ndarray:
    """Robustly estimate the point cloud center.

    Args:
        point_cloud: Array of shape (N,3) representing 3d points.

    Returns:
        mean_pt: Coordinates of central point, ignoring outliers.
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
        args: Rendering options.
    """
    # Read in data.
    wTi_list, img_fnames, calibrations, point_cloud, rgb = io_utils.read_scene_data_from_colmap_format(
        data_dir=args.output_dir, file_format=args.file_format
    )
    if args.show_mvs_result:
        point_cloud, rgb = io_utils.read_point_cloud_from_ply(args.ply_fpath)

    if len(calibrations) == 1:
        calibrations = calibrations * len(img_fnames)
    mean_pt = compute_point_cloud_center_robust(point_cloud)

    # Zero-center the point cloud (about estimated center).
    zcwTw = Pose3(Rot3(np.eye(3)), -mean_pt)
    # expression below is equivalent to applying zcwTw.transformFrom() to each world point
    point_cloud -= mean_pt
    is_nearby = np.linalg.norm(point_cloud, axis=1) < args.max_range
    point_cloud = point_cloud[is_nearby]
    rgb = rgb[is_nearby]
    for i in range(len(wTi_list)):
        wTi_list[i] = zcwTw.compose(wTi_list[i])

    draw_scene_open3d(point_cloud, rgb, wTi_list, calibrations, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize GTSFM result with Open3d.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "ba_output"),
        help="Path to a directory containing GTSFM output. "
        "This directory should contain 3 files: either `cameras.txt`, `images.txt`, and `points3D.txt`"
        " or `cameras.bin`, `images.bin`, and `points3D.bin`.",
    )
    parser.add_argument(
        "--file_format",
        type=str,
        default="txt",
        help="File format for data in `output_dir`, indicating if stored as text (txt) or binary data (bin).",
    )
    parser.add_argument(
        "--rendering_style",
        type=str,
        default="point",
        choices=["point", "sphere"],
        help="Render each 3d point as a `point` (optimized in Open3d) or `sphere` (unoptimized in Open3d).",
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
    parser.add_argument(
        "--frustum_ray_len",
        type=float,
        default=0.3,
        help="Length to extend frustum rays away from optical center "
        + "(increase length for large-scale scenes to make frustums visible)",
    )
    parser.add_argument(
        "--show_mvs_result",
        action="store_true",
        help="defaults to false.",
    )
    parser.add_argument(
        "--ply_fpath",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "mvs_output", "dense_pointcloud.ply"),
        help="Path to MVS output (.ply file).",
    )

    args = parser.parse_args()
    view_scene(args)
