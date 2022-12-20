"""Functions to create transforms.json file and resized images that are input to Nerfstudio

Here I have modified code taken from Nerfstudio for parsing data in the Nerfstudio format.
Original files at:
https://github.com/nerfstudio-project/nerfstudio/blob/main/scripts/process_data.py
https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/process_data/colmap_utils.py

Author: Jon Womack
"""
import argparse
import json
import os
from enum import Enum
from pathlib import Path

import numpy as np
import thirdparty.nerfstudio.colmap_utils as colmap_utils

from gtsfm.utils import images, io


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"


CAMERA_MODELS = {
    "perspective": CameraModel.OPENCV,
    "fisheye": CameraModel.OPENCV_FISHEYE,
}


def colmap_to_json(cameras_path: Path, images_path: Path, output_dir: Path, camera_model: CameraModel) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.
    Args:
        cameras_path: Path to the cameras.txt file.
        images_path: Path to the images.txt file.
        output_dir: Path to the output directory.
        camera_model: Camera model used.
    Returns:
        The number of registered images.
    """

    cameras = colmap_utils.read_cameras_text(cameras_path)
    images = colmap_utils.read_images_text(images_path)

    # Only supports one camera
    camera_params = cameras[1].params

    frames = []
    for _, im_data in images.items():
        rotation = colmap_utils.qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system to ours
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = Path(f"./images/{im_data.name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    out = {
        "fl_x": float(camera_params[0]),
        "fl_y": float(camera_params[1]),
        "cx": float(camera_params[2]),
        "cy": float(camera_params[3]),
        "w": cameras[1].width,
        "h": cameras[1].height,
        "camera_model": camera_model.value,
    }

    if camera_model == CameraModel.OPENCV:
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "p1": float(camera_params[6]),
                "p2": float(camera_params[7]),
            }
        )
    if camera_model == CameraModel.OPENCV_FISHEYE:
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "k3": float(camera_params[6]),
                "k4": float(camera_params[7]),
            }
        )

    out["frames"] = frames

    with open(os.path.join(output_dir, "transforms.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return cameras[1].width, cameras[1].height


def resize_and_save_images_for_nerfstudio(images_dir, image_width, image_height, resized_images_dir):
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
        image = io.load_image(image_path)
        resized_image = images.resize_image(image, image_height, image_width)
        resized_image_path = os.path.join(resized_images_dir, image_filename)
        io.save_image(resized_image, resized_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=True, help="Relative or absolute path to /gtsfm/results/")
    parser.add_argument(
        "--camera_model",
        required=True,
        help="'perspective' or 'fisheye' corresponding to the OPENCV and OPENCV_FISHEYE camera models at" 
        + " https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/cameras.py",
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Relative or absolute path to the directory of images GTSfM used for reconstruction",
    )
    args = parser.parse_args()

    # Create transforms.json, which contains intrinsics and extrinsics
    results_path = args.results_path
    cameras_file = os.path.join(results_path, "ba_output", "cameras.txt")
    images_file = os.path.join(results_path, "ba_output", "images.txt")
    nerfstudio_input_dir = os.path.join(results_path, "nerfstudio-input")
    if not os.path.exists(nerfstudio_input_dir):
        os.makedirs(nerfstudio_input_dir)
    image_width, image_height = colmap_to_json(
        cameras_file, images_file, nerfstudio_input_dir, camera_model=CAMERA_MODELS[args.camera_model]
    )

    # Resize images (as GTSfM does) to match intrinsics in transforms.json
    gtsfm_images_dir = args.images_dir
    nerfstudio_images_dir = os.path.join(nerfstudio_input_dir, "images")
    if not os.path.exists(nerfstudio_images_dir):
        os.makedirs(nerfstudio_images_dir)
    resize_and_save_images_for_nerfstudio(gtsfm_images_dir, image_width, image_height, nerfstudio_images_dir)
