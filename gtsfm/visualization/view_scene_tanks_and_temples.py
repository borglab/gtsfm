"""Script to render Tanks & Temples dataset ground truth camera poses and structure, and render the scene to the GUI.

Results must be stored in the Tanks & Temples file format.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import open3d

import gtsfm.utils.io as io_utils
import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils
from gtsfm.loader.tanks_and_temples_loader import TanksAndTemplesLoader

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "tests" / "data"


def view_scene(args: argparse.Namespace) -> None:
    """Read Tanks & Temples Dataset ground truth from JSON, txt, PLY, and .log files and render the scene to the GUI.

    Args:
        args: Rendering options.
    """
    data_root = Path(args.data_root)
    img_dir = data_root / args.scene_name
    poses_fpath = data_root / f"{args.scene_name}_COLMAP_SfM.log"
    ply_alignment_fpath = data_root / f"{args.scene_name}_trans.txt"
    bounding_polyhedron_json_fpath = data_root / f"{args.scene_name}.json"

    lidar_ply_fpath = data_root / f"{args.scene_name}.ply"
    colmap_ply_fpath = data_root / f"{args.scene_name}_COLMAP.ply"

    # Note: PLY files are not provided here, as they are too large to include as test data (300 MB each).
    loader = TanksAndTemplesLoader(
        img_dir=str(img_dir),
        poses_fpath=str(poses_fpath),
        bounding_polyhedron_json_fpath=str(bounding_polyhedron_json_fpath),
        ply_alignment_fpath=str(ply_alignment_fpath),
        lidar_ply_fpath=str(lidar_ply_fpath),
        colmap_ply_fpath=str(colmap_ply_fpath),
    )

    # Both are loaded in the COLMAP world coordinate frame.
    lidar_pcd = loader.get_lidar_point_cloud(downsample_factor=1000)
    overlay_point_clouds = False
    if overlay_point_clouds:
        red = np.zeros_like(lidar_pcd.colors)
        red[:, 0] = 1.0
        lidar_pcd.colors = open3d.utility.Vector3dVector(red)

    colmap_pcd = loader.get_colmap_point_cloud(downsample_factor=100)

    calibrations = [loader.get_camera_intrinsics_full_res(0)] * len(loader)
    wTi_list = loader.get_gt_poses()
    frustums = open3d_vis_utils.create_all_frustums_open3d(wTi_list, calibrations, args.frustum_ray_len)

    geometries = frustums + [colmap_pcd] + [lidar_pcd]
    open3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Tanks & Temples dataset w/ Open3d.")
    parser.add_argument(
        "--data_root", type=str, default=os.path.join(TEST_DATA_ROOT, "tanks_and_temples_barn"), help=""
    )
    parser.add_argument("--scene_name", type=str, default="Barn")
    parser.add_argument(
        "--frustum_ray_len",
        type=float,
        default=0.1,
        help="Length to extend frustum rays away from optical center "
        + "(increase length for large-scale scenes to make frustums visible)",
    )
    args = parser.parse_args()
    view_scene(args)
