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
        args: Rendering options.
    """
    loader = OlssonLoader(
        args.dataset_root,
        max_frame_lookahead=DUMMY_MAX_FRAME_LOOKAHEAD,
        max_resolution=args.max_resolution,
    )

    point_cloud_rgb = np.zeros(shape=loader._point_cloud.shape, dtype=np.uint8)
    derive_point_colors = True
    if derive_point_colors:

        tracks = loader.gt_tracks

        cameras = {i:loader.get_camera(i) for i in range(len(loader))}
        images = {i:loader.get_image(i) for i in range(len(loader))}

        for j, track in enumerate(tracks):
            if track.numberMeasurements() == 0:
                continue
            track_colors = []
            # Cannot naively project 3d point into images since we do not know occlusion info.
            point = track.point3()
            for k in range(track.numberMeasurements()):
                i, uv = track.measurement(k)
                # uv_reprojected, success_flag = cameras[i].projectSafe(point)
                # if not success_flag:
                #     import pdb; pdb.set_trace()
                u, v = uv.astype(np.int32)
                track_colors.append(images[i].value_array[v, u])
            avg_color = np.array(track_colors).mean(axis=0)
            point_cloud_rgb[j] = avg_color
        # for j, point in enumerate(loader._point_cloud):

        #     # Have to use track to get visibility info.
        #     track_colors = []
        #     print(f"On point j={j}")
        #     for i in range(len(loader)):
        #         if u < 0 or u >= images[i].width or v < 0 or v >= images[i].height:
        #             continue

    open3d_vis_utils.draw_scene_open3d(
        point_cloud=loader._point_cloud,
        rgb=point_cloud_rgb,
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
    parser.add_argument(
        "--derive_point_colors",
        action="store_true",
        help="Derive RGB point colors by projecting each 3D point into images (slow)."
    )
    args = parser.parse_args()
    view_scene(args)
