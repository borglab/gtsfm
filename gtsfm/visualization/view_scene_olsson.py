"""
Script to render Olsson dataset ground truth camera poses and structure, and render the scene to the GUI.

Results must be stored in Carl Olsson's `data.mat` file format.

Authors: John Lambert
"""
import argparse
import os
from pathlib import Path

import numpy as np

import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils
from gtsfm.loader.olsson_loader import OlssonLoader

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "tests" / "data"

# The Olsson loader requires we set a value for the frame lookahead, but it will not affect the visualization.
DUMMY_MAX_FRAME_LOOKAHEAD = 1


def view_scene(args: argparse.Namespace) -> None:
    """Read Olsson Dataset ground truth from a data.mat file and render the scene to the GUI.

    Args:
        args: rendering options.
    """
    loader = OlssonLoader(
        args.dataset_root,
        max_frame_lookahead=DUMMY_MAX_FRAME_LOOKAHEAD,
        max_resolution=args.max_resolution,
    )
    open3d_vis_utils.draw_scene_open3d(
        point_cloud=loader._point_cloud,
        rgb=np.ones_like(loader._point_cloud).astype(np.uint8),
        wTi_list=loader._wTi_list,
        calibrations=[loader.get_camera_intrinsics_full_res(0)] * loader._num_imgs,
        args=args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Olsson dataset w/ Open3d.")
    parser.add_argument("--dataset_root", type=str, default=os.path.join(TEST_DATA_ROOT, "set1_lund_door"), help="")
    parser.add_argument(
        "--rendering_style",
        type=str,
        default="point",
        choices=["point", "sphere"],
        help="Render each 3d point as a `point` (optimized in Open3d) or `sphere` (optimized in Mayavi).",
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
        "--max_resolution",
        type=int,
        default=760,
        help="integer representing maximum length of image's short side"
        " e.g. for 1080p (1920 x 1080), max_resolution would be 1080",
    )
    args = parser.parse_args()
    view_scene(args)
