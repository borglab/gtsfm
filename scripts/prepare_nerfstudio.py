"""Creates transforms.json file and resized images for running Nerfstudio from gtsfm output.

This contains code from Nerfstudio for parsing data into the Nerfstudio format.
https://github.com/nerfstudio-project/nerfstudio/blob/a121d76fd085f0fb356abf150c089c42eecbd066/nerfstudio/process_data/colmap_converter_to_nerfstudio_dataset.py#L29
https://github.com/nerfstudio-project/nerfstudio/blob/a121d76fd085f0fb356abf150c089c42eecbd066/nerfstudio/process_data/colmap_utils.py#L16

Author: Jon Womack
"""
import argparse
import json
import os
from enum import Enum
from pathlib import Path
from typing import Tuple

import numpy as np
from gtsam import Rot3

from gtsfm.utils import images
from gtsfm.utils import io as io_utils


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"


CAMERA_MODELS = {
    "perspective": CameraModel.OPENCV,
    "fisheye": CameraModel.OPENCV_FISHEYE,
}


def colmap_to_json(data_dir: str, output_dir: str, camera_model: CameraModel) -> Tuple[int, int]:
    """Converts GTSfM's cameras.txt to to a JSON file, saves it in output_dir.
    Args:
        cameras_path: Path to the cameras.txt file.
        images_path: Path to the images.txt file.
        output_dir: Path to the output directory.
        camera_model: Camera model used.
    Returns:
        Image dimension as a tuple.
    """
    wTi_list, input_images, calibrations, _, _, img_dims = io_utils.read_scene_data_from_colmap_format(data_dir)

    frames = []
    for i, image_name in enumerate(input_images):
        c2w = wTi_list[i].matrix()

        # Convert from GTSfM coordinate frame to nerfstudio coordinate frame.
        # Invert camera X and Y, swap X and Y and invert Z in the world frame.
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = Path(f"./images/{image_name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    # Only supports single intrinsics datasets.
    camera_params = calibrations[0]
    img_h, img_w = img_dims[0]

    out = {
        "fl_x": float(camera_params.fx()),
        "fl_y": float(camera_params.fy()),
        "cx": float(camera_params.px()),
        "cy": float(camera_params.py()),
        "w": img_w,
        "h": img_h,
        "camera_model": camera_model.value,
    }

    if camera_model == CameraModel.OPENCV:
        out.update(
            {
                "k1": float(camera_params.k1()),
                "k2": float(camera_params.k2()),
                "p1": 0.0,
                "p2": 0.0,
            }
        )
    if camera_model == CameraModel.OPENCV_FISHEYE:
        out.update(
            {
                "k1": camera_params.k1(),
                "k2": camera_params.k2(),
                "k3": 0.0,
                "k4": 0.0,
            }
        )

    out["frames"] = frames

    with open(os.path.join(output_dir, "transforms.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return img_w, img_h


def resize_and_save_images_for_nerfstudio(
    images_dir: str, image_width: int, image_height: int, resized_images_dir: str
) -> None:
    """Resizes images to the size used in GTSfM and saves for nerfstudio.
    Args:
        images_dir: path to the original images.
        images_width: int width of resized images
        images_height: int height of resized images.
        resized_images_dir: path to directory where the resized images will be saved.

    """
    image_filenames = os.listdir(images_dir)
    for image_filename in image_filenames:
        image_path = os.path.join(images_dir, image_filename)
        image = io_utils.load_image(image_path)
        resized_image = images.resize_image(image, image_height, image_width)
        resized_image_path = os.path.join(resized_images_dir, image_filename)
        io_utils.save_image(resized_image, resized_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=True, help="Path to /gtsfm/results/, should contain ba_output/")
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Path to the directory of images GTSfM used for reconstruction",
    )
    parser.add_argument(
        "--camera_model",
        default="perspective",
        help="'perspective' or 'fisheye' corresponding to the OPENCV and OPENCV_FISHEYE camera models at"
        + " https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/cameras.py",
    )
    args = parser.parse_args()

    # Create transforms.json, which contains intrinsics and extrinsics
    results_path = args.results_path
    cameras_file = os.path.join(results_path, "ba_output", "cameras.txt")
    images_file = os.path.join(results_path, "ba_output", "images.txt")
    nerfstudio_input_dir = os.path.join(results_path, "nerfstudio_input")
    data_dir = os.path.join(results_path, "ba_output")
    if not os.path.exists(nerfstudio_input_dir):
        os.makedirs(nerfstudio_input_dir)
    image_width, image_height = colmap_to_json(
        data_dir, nerfstudio_input_dir, camera_model=CAMERA_MODELS[args.camera_model]
    )

    # Resize images (as GTSfM does) to match intrinsics in transforms.json
    gtsfm_images_dir = args.images_dir
    nerfstudio_images_dir = os.path.join(nerfstudio_input_dir, "images")
    if not os.path.exists(nerfstudio_images_dir):
        os.makedirs(nerfstudio_images_dir)
    resize_and_save_images_for_nerfstudio(gtsfm_images_dir, image_width, image_height, nerfstudio_images_dir)
