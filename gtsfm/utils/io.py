"""Functions to provide I/O APIs for all the modules.

Authors: Ayush Baid, John Lambert
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gtsam
import h5py
import json
import numpy as np
from gtsam import Cal3Bundler, Rot3, Pose3
from PIL import Image as PILImage
from PIL.ExifTags import GPSTAGS, TAGS

import gtsfm.utils.images as image_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.reprojection as reproj_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.sfm_track import SfmTrack2d


logger = logger_utils.get_logger()


def load_image(img_path: str) -> Image:
    """Load the image from disk.

    Note: EXIF is read as a map from (tag_id, value) where tag_id is an integer.
    In order to extract human-readable names, we use the lookup table TAGS or GPSTAGS.

    Args:
        img_path (str): the path of image to load.

    Returns:
        loaded image in RGB format.
    """
    original_image = PILImage.open(img_path)

    exif_data = original_image._getexif()
    if exif_data is not None:
        parsed_data = {}
        for tag_id, value in exif_data.items():
            # extract the human readable tag name
            if tag_id in TAGS:
                tag_name = TAGS.get(tag_id)
            elif tag_id in GPSTAGS:
                tag_name = GPSTAGS.get(tag_id)
            else:
                tag_name = tag_id
            parsed_data[tag_name] = value

        exif_data = parsed_data

    img_fname = Path(img_path).name
    return Image(value_array=np.asarray(original_image.convert('RGB')), exif_data=exif_data, file_name=img_fname)


def save_image(image: Image, img_path: str) -> None:
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


def save_json_file(
    json_fpath: str,
    data: Union[Dict[Any, Any], List[Any]],
) -> None:
    """Save a Python dictionary or list to a JSON file.
    Args:
        json_fpath: Path to file to create.
        data: Python dictionary or list to be serialized.
    """
    os.makedirs(os.path.dirname(json_fpath), exist_ok=True)
    with open(json_fpath, "w") as f:
        json.dump(data, f, indent=4)


def read_json_file(fpath: Union[str, Path]) -> Any:
    """Load dictionary from JSON file.

    Args:
        fpath: Path to JSON file.

    Returns:
        Deserialized Python dictionary or list.
    """
    with open(fpath, "r") as f:
        return json.load(f)


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


def export_model_as_colmap_text(gtsfm_data: GtsfmData, images: List[Image], save_dir: str) -> None:
    """Emulates the COLMAP option to `Export model as text`.

    Three text files will be save to disk: "points3D.txt", "images.txt", and "cameras.txt".

    Args:
        gtsfm_data: scene data to write.
        images: list of all images for this scene, in order of image index.
        save_dir: folder where text files will be saved.
    """
    write_cameras(gtsfm_data, images, save_dir)
    write_images(gtsfm_data, images, save_dir)
    write_points(gtsfm_data, images, save_dir)


def read_cameras_txt(fpath: str) -> Optional[List[Cal3Bundler]]:
    """Read camera calibrations from a COLMAP-formatted cameras.txt file.

    Reference: https://colmap.github.io/format.html#cameras-txt

    Args:
        fpaths: path to cameras.txt file

    Returns:
        calibration object for each camera, or None if requested file is non-existent
    """
    if not Path(fpath).exists():
        logger.info("%s does not exist", fpath)
        return None

    with open(fpath, "r") as f:
        lines = f.readlines()

    # may not be one line per camera (could be only one line of text if shared calibration)
    num_cams = int(lines[2].replace("# Number of cameras: ", "").strip())

    calibrations = []
    for line in lines[3:]:

        cam_params = line.split()
        # Note that u0 is px, and v0 is py
        cam_id, model, img_w, img_h, fx, u0, v0 = cam_params[:7]
        img_w, img_h, fx, u0, v0 = int(img_w), int(img_h), float(fx), float(u0), float(v0)
        # TODO: determine convention for storing/reading radial distortion parameters
        k1 = 0
        k2 = 0
        calibrations.append(Cal3Bundler(fx, k1, k2, u0, v0))

    #assert len(calibrations) == num_cams
    return calibrations


def write_cameras(gtsfm_data: GtsfmData, images: List[Image], save_dir: str) -> None:
    """Writes the camera data file in the COLMAP format.

    Reference: https://colmap.github.io/format.html#cameras-txt

    Args:
        gtsfm_data: scene data to write.
        images: list of all images for this scene, in order of image index
        save_dir: folder to put the cameras.txt file in.
    """
    os.makedirs(save_dir, exist_ok=True)

    # TODO: handle shared intrinsics
    camera_model = "SIMPLE_RADIAL"

    file_path = os.path.join(save_dir, "cameras.txt")
    with open(file_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {gtsfm_data.number_images()}\n")

        for i in gtsfm_data.get_valid_camera_indices():
            camera = gtsfm_data.get_camera(i)
            calibration = camera.calibration()

            fx = calibration.fx()
            u0 = calibration.px()
            v0 = calibration.py()
            k1 = calibration.k1()
            k2 = calibration.k2()

            image_height = images[i].height
            image_width = images[i].width

            f.write(f"{i} {camera_model} {image_width} {image_height} {fx} {u0} {v0} {k1} {k2}\n")


def read_images_txt(fpath: str) -> Tuple[Optional[List[Pose3]], Optional[List[str]]]:
    """Read camera poses and image file names from a COLMAP-format images.txt file.

    Reference: https://colmap.github.io/format.html#images-txt
        "The coordinates of the projection/camera center are given by -R^t * T, where
        R^t is the inverse/transpose of the 3x3 rotation matrix composed from the
        quaternion and T is the translation vector. The local camera coordinate system
        of an image is defined in a way that the X axis points to the right, the Y axis
        to the bottom, and the Z axis to the front as seen from the image."

    Args:
        fpath: path to images.txt file

    Returns:
        wTi_list: list of camera poses for each image, or None if file path invalid
        img_fnames: name of image file, for each image, or None if file path invalid
    """
    if not Path(fpath).exists():
        logger.info("%s does not exist", fpath)
        return None, None

    with open(fpath, "r") as f:
        lines = f.readlines()

    wTi_list = []
    img_fnames = []
    # ignore first 4 lines of text -- they are a description of the file format
    for line in lines[4::2]:
        i, qw, qx, qy, qz, tx, ty, tz, i, img_fname = line.split()
        # Colmap provides extrinsics, so must invert
        iRw = Rot3(float(qw), float(qx), float(qy), float(qz))
        wTi = Pose3(iRw, np.array([tx, ty, tz], dtype=np.float64)).inverse()
        wTi_list.append(wTi)
        img_fnames.append(img_fname)

    return wTi_list, img_fnames


def write_images(gtsfm_data: GtsfmData, images: List[Image], save_dir: str) -> None:
    """Writes the image data file in the COLMAP format.

    Reference: https://colmap.github.io/format.html#images-txt

    Args:
        gtsfm_data: scene data to write.
        images: list of all images for this scene, in order of image index.
        save_dir: folder to put the images.txt file in.
    """
    os.makedirs(save_dir, exist_ok=True)

    num_imgs = gtsfm_data.number_images()
    # TODO: compute this (from keypoint data? or from track data?)
    mean_obs_per_img = 0

    file_path = os.path.join(save_dir, "images.txt")
    with open(file_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {num_imgs}, mean observations per image: {mean_obs_per_img}\n")

        for i in gtsfm_data.get_valid_camera_indices():
            img_fname = images[i].file_name
            camera = gtsfm_data.get_camera(i)
            # COLMAP exports camera extrinsics (cTw), not the poses (wTc), so must invert
            iTw = camera.pose().inverse()
            iRw_quaternion = iTw.rotation().quaternion()
            itw = iTw.translation()
            tx, ty, tz = itw
            qw, qx, qy, qz = iRw_quaternion

            f.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {img_fname}\n")
            # TODO: write out the points2d
            f.write("TODO\n")


def read_points_txt(fpath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Read 3d points and their associated colors from a COLMAP points.txt file.

    Reference: https://colmap.github.io/format.html#points3d-txt

    Args:
        fpath: absolute file path to points.txt file

    Returns:
        point_cloud: float array of shape (N,3)
        rgb: uint8 array of shape (N,3)
    """
    if not Path(fpath).exists():
        logger.info("%s does not exist", fpath)
        return None, None

    with open(fpath, "r") as f:
        data = f.readlines()

    rgb = []
    point_cloud = []
    # first 3 lines are information about the file format
    # line at index 2 will be of the form
    # "# Number of points: 2122, mean track length: 2.8449575871819039"
    points_metadata = data[2]
    j = points_metadata.find(":")
    k = points_metadata.find(",")
    expected_num_pts = int(points_metadata[j + 1 : k])

    data = data[3:]
    for line in data:
        entries = line.split()
        x, y, z, r, g, b = entries[1:7]

        point = [float(x), float(y), float(z)]
        point_cloud += [point]
        rgb += [(int(r), int(g), int(b))]

    point_cloud = np.array(point_cloud)
    rgb = np.array(rgb).astype(np.uint8)

    assert point_cloud.shape[0] == expected_num_pts
    assert rgb.shape[0] == expected_num_pts
    return point_cloud, rgb


def write_points(gtsfm_data: GtsfmData, images: List[Image], save_dir: str) -> None:
    """Writes the point cloud data file in the COLMAP format.

    Reference: https://colmap.github.io/format.html#points3d-txt

    Args:
        gtsfm_data: scene data to write.
        images: list of all images for this scene, in order of image index
        save_dir: folder to put the points3D.txt file in.
    """
    os.makedirs(save_dir, exist_ok=True)

    num_pts = gtsfm_data.number_tracks()
    avg_track_length, _ = gtsfm_data.get_track_length_statistics()

    file_path = os.path.join(save_dir, "points3D.txt")
    with open(file_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {num_pts}, mean track length: {np.round(avg_track_length, 2)}\n")

        # TODO: assign unique indices to all keypoints (2d points)
        point2d_idx = 0

        for j in range(num_pts):
            track = gtsfm_data.get_track(j)

            r, g, b = image_utils.get_average_point_color(track, images)
            _, avg_track_reproj_error = reproj_utils.compute_track_reprojection_errors(gtsfm_data._cameras, track)
            x, y, z = track.point3()
            f.write(f"{j} {x} {y} {z} {r} {g} {b} {np.round(avg_track_reproj_error, 2)} ")

            for k in range(track.number_measurements()):
                i, uv_measured = track.measurement(k)
                f.write(f"{i} {point2d_idx} ")
            f.write("\n")


def save_track_visualizations(
    tracks_2d: List[SfmTrack2d],
    images: List[Image],
    save_dir: str,
    viz_patch_sz: int = 100,
) -> None:
    """For every track, save an image with vertically stacked patches, each corresponding to a track keypoint.

    The visualizations can serve as a useful debugging tool for finding erroneous matches within tracks.
    """
    os.makedirs(save_dir, exist_ok=True)

    # save each 2d track
    for i, track in enumerate(tracks_2d):
        patches = []
        for m in track.measurements:
            patches += [images[m.i].extract_patch(center_x=m.uv[0], center_y=m.uv[1], patch_size=viz_patch_sz)]

        stacked_image = image_utils.vstack_image_list(patches)
        save_fpath = os.path.join(save_dir, f"track_{i}.jpg")
        save_image(stacked_image, img_path=save_fpath)
