"""Functions to provide I/O APIs for all the modules.

Authors: Ayush Baid
"""
import os
from typing import Any, Dict, List, Union

import gtsam
import h5py
import json
import numpy as np
from PIL import Image as PILImage
from PIL.ExifTags import GPSTAGS, TAGS

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image


def load_image(img_path: str) -> Image:
    """Load the image from disk.

    Args:
        img_path (str): the path of image to load.

    Returns:
        loaded image in RGB format.
    """
    original_image = PILImage.open(img_path)

    exif_data = original_image.getexif()
    if exif_data is not None:
        parsed_data = {}
        for tag, value in exif_data.items():
            if tag in TAGS:
                parsed_data[TAGS.get(tag)] = value
            elif tag in GPSTAGS:
                parsed_data[GPSTAGS.get(tag)] = value
            else:
                parsed_data[tag] = value

        exif_data = parsed_data

    return Image(np.asarray(original_image), exif_data)


def save_image(image: Image, img_path: str):
    """Saves the image to disk

    Args:
        image (np.array): image
        img_path (str): the path on disk to save the image to
    """
    im = PILImage.fromarray(image.value_array)
    im.save(img_path)


def load_h5(file_path: str) -> Dict[Any, Any]:
    """Loads a dictionary from a h5 file

    Args:
        file_path: path of the h5 file

    Returns:
        the dictionary from the h5 file
    """

    data = {}

    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            data[key] = f[key][:]

    return data


def save_json_file(json_fpath: str, data: Union[Dict[Any, Any], List[Any]],) -> None:
    """Save a Python dictionary or list to a JSON file.
    Args:
        json_fpath: Path to file to create.
        data: Python dictionary or list to be serialized.
    """
    os.makedirs(os.path.dirname(json_fpath), exist_ok=True)
    with open(json_fpath, "w") as f:
        json.dump(data, f, indent=4)


def read_bal(file_path: str) -> GtsfmData:
    """Read a Bundle Adjustment in the Large" (BAL) file.
    
    See https://grail.cs.washington.edu/projects/bal/ for more details on the format.


    Args:
        file_name: file path of the BAL file.

    Returns:
        The data as an GtsfmData object.
    """
    sfm_data = gtsam.readBal(file_path)

    num_images = sfm_data.number_cameras()

    gtsfm_data = GtsfmData(num_images)
    for i in range(num_images):
        camera = sfm_data.camera(i)
        gtsfm_data.add_camera(i, camera)
    for j in range(sfm_data.number_tracks()):
        gtsfm_data.add_track(sfm_data.track(j))

    return gtsfm_data


def write_cameras(gtsfm_data: GtsfmData, save_dir: str) -> None:
    """Writes the camera data file in the COLMAP format.

    Args:
        gtsfm_data: scene data to write.
        save_dir: folder to put the cameras.txt file in.
    """
    os.makedirs(save_dir, exist_ok=True)

    # TODO: get image shape somehow
    image_width = 1000
    image_height = 1000

    # TODO: handle shared intrinsics
    camera_model = "SIMPLE_RADIAL"

    file_path = os.path.join(save_dir, "cameras.txt")
    with open(file_path, "w") as f:
        f.write("# Camera list with one line of data per camera:")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]")
        f.write(f"# Number of cameras: {gtsfm_data.number_images()}\n")

        for i in gtsfm_data.get_valid_camera_indices():
            camera = gtsfm_data.get_camera(i)
            calibration = camera.calibration()

            fx = calibration.fx()
            u0 = calibration.px()
            v0 = calibration.py()
            k1 = calibration.k1()
            k2 = calibration.k2()

            f.write(f"{i} {camera_model} {image_width} {image_height} {fx} {u0} {v0} {k1} {k2}\n")


def write_images(gtsfm_data: GtsfmData, save_dir: str) -> None:
    """Writes the image data file in the COLMAP format.

    Args:
        gtsfm_data: scene data to write.
        save_dir: folder to put the images.txt file in.
    """
    os.makedirs(save_dir, exist_ok=True)

    num_imgs = gtsfm_data.number_images()
    # TODO: compute this
    mean_obs_per_img = 0

    #TODO: compute this
    img_fname = "dummy.jpg"

    file_path = os.path.join(save_dir, "images.txt")
    with open(file_path, "w") as f:
        f.write("# Image list with two lines of data per image:")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)")
        f.write(f"# Number of cameras: {num_imgs}, mean observations per image: {mean_obs_per_img}\n")

        for i in gtsfm_data.get_valid_camera_indices():
            camera = gtsfm_data.get_camera(i)
            wRi_quaternion = camera.pose().rotation().quaternion()
            wti = camera.pose().translation()
            tx, ty, tz = wti
            qw, qx, qy, qz = wRi_quaternion

            f.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {img_fname}\n")
            # TODO: write out the points2d


def write_points(gtsfm_data: GtsfmData, save_dir: str) -> None:
    """Writes the point cloud data file in the COLMAP format.

    Args:
        gtsfm_data: scene data to write.
        save_dir: folder to put the points3D.txt file in.
    """
    os.makedirs(save_dir, exist_ok=True)

    num_pts = gtsfm_data.number_tracks()
    avg_track_length, _, _ = gtsfm_data.get_track_length_statistics()

    file_path = os.path.join(save_dir, "point3D.txt")
    with open(file_path, "w") as f:
        f.write("# Number of cameras: {}\n".format(gtsfm_data.number_images()))

        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {num_pts}, mean track length: {avg_track_length}\n")

        # TODO: extract point color
        r, g, b = 0, 0, 0
        # TODO: coming in another PR
        track_reproj_error = 0
        # TODO: assign unique indices to all keypoints (2d points)
        point2d_idx = 0

        for j in range(num_pts):
            track = gtsfm_data.get_track(j)
            x,y,z = track.point3()
            f.write(f"{j}, {x}, {y}, {z}, {r}, {g}, {b}, {track_reproj_error}"

            for k in range(track.number_measurements()):
                i, uv_measured = track.measurement(k)
                f.write(f"{i} {point2d_idx}")
            f.write("\n")

    