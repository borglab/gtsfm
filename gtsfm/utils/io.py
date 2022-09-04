"""Functions to provide I/O APIs for all the modules.

Authors: Ayush Baid, John Lambert
"""
import os
import pickle
from bz2 import BZ2File
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gtsam
import h5py
import numpy as np
import open3d
import simplejson as json
from gtsam import Cal3Bundler, Point3, Pose3, Rot3, SfmTrack
from PIL import Image as PILImage
from PIL.ExifTags import GPSTAGS, TAGS

import gtsfm.utils.images as image_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.reprojection as reproj_utils
import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.sfm_track import SfmTrack2d
from thirdparty.colmap.scripts.python.read_write_model import Camera as ColmapCamera
from thirdparty.colmap.scripts.python.read_write_model import Image as ColmapImage
from thirdparty.colmap.scripts.python.read_write_model import Point3D as ColmapPoint3D

logger = logger_utils.get_logger()


def load_image(img_path: str) -> Image:
    """Load the image from disk.

    Notes: EXIF is read as a map from (tag_id, value) where tag_id is an integer.
    In order to extract human-readable names, we use the lookup table TAGS or GPSTAGS.
    Images will be converted to RGB if in a different format.

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
    original_image = original_image.convert("RGB") if original_image.mode != "RGB" else original_image
    return Image(value_array=np.asarray(original_image), exif_data=exif_data, file_name=img_fname)


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
        # ignore_nan=False replaces any NaN with null so that RTF frontend can
        # parse it
        json.dump(data, f, indent=4, ignore_nan=True)


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
    return GtsfmData.from_sfm_data(sfm_data)


def read_bundler(file_path: str) -> GtsfmData:
    """Read a Bundler file.

    Args:
        file_name: file path of the Bundler file.

    Returns:
        The data as an GtsfmData object.
    """
    sfm_data = gtsam.SfmData.FromBundlerFile(file_path)
    return GtsfmData.from_sfm_data(sfm_data)


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


def colmap2gtsfm(
    cameras: Dict[int, ColmapCamera],
    images: Dict[int, ColmapImage],
    points3D: Dict[int, ColmapPoint3D],
    load_sfmtracks: bool = False,
) -> Tuple[List[Cal3Bundler], List[Pose3], List[str], Optional[List[Point3]]]:
    """Converts COLMAP-formatted variables to GTSfM format.
    Args:
        cameras: dictionary of COLMAP-formatted Cameras
        images: dictionary of COLMAP-formatted Images
        points3D: dictionary of COLMAP-formatted Point3Ds
        return_tracks (optional): whether or not to return tracks
    Returns:
        img_fnames: file names of images in images_gtsfm
        images_gtsfm: list of N camera poses when each image was taken
        cameras_gtsfm: list of N camera calibrations corresponding to the N images in images_gtsfm
        sfmtracks_gtsfm: tracks of points in points3D
    """
    # Note: Assumes input cameras use `PINHOLE` model
    if len(images) == 0 and len(cameras) == 0:
        raise RuntimeError("No Image or Camera data provided to loader.")
    cameras_gtsfm, images_gtsfm, img_fnames = [], [], []
    image_id_to_idx = {}  # keeps track of discrepencies between `image_id` and List index.
    for idx, img in enumerate(images.values()):
        images_gtsfm.append(Pose3(Rot3(img.qvec2rotmat()), img.tvec).inverse())
        img_fnames.append(img.name)
        fx, _, cx, cy = cameras[img.camera_id].params[:4]
        cameras_gtsfm.append(Cal3Bundler(fx, 0.0, 0.0, cx, cy))
        image_id_to_idx[img.id] = idx

    if len(points3D) == 0 and load_sfmtracks:
        raise RuntimeError("No SfMTrack data provided to loader.")
    sfmtracks_gtsfm = None
    if len(points3D) > 0 and load_sfmtracks:
        sfmtracks_gtsfm = []
        for point3D in points3D.values():
            sfmtrack = SfmTrack(point3D.xyz)
            for (image_id, point2d_idx) in zip(point3D.image_ids, point3D.point2D_idxs):
                sfmtrack.addMeasurement(image_id_to_idx[image_id], images[image_id].xys[point2d_idx])
            sfmtracks_gtsfm.append(sfmtrack)

    return img_fnames, images_gtsfm, cameras_gtsfm, sfmtracks_gtsfm


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
        model = cam_params[1]
        # Currently only handles SIMPLE RADIAL and RADIAL camera models
        assert model in ["SIMPLE_RADIAL", "RADIAL"]
        if model == "SIMPLE_RADIAL":
            _, _, img_w, img_h, fx, u0, v0, k1 = cam_params[:8]
            img_w, img_h, fx, u0, v0, k1 = int(img_w), int(img_h), float(fx), float(u0), float(v0), float(k1)
            # Convert COLMAP's SIMPLE_RADIAL to GTSAM's Cal3Bundler:
            # Add second radial distortion coefficient of value zero.
            k2 = 0
            calibrations.append(Cal3Bundler(fx, k1, k2, u0, v0))
        elif model == "RADIAL":
            _, _, img_w, img_h, fx, u0, v0, k1, k2 = cam_params[:9]
            img_w, img_h, fx, u0, v0, k1, k2 = (
                int(img_w),
                int(img_h),
                float(fx),
                float(u0),
                float(v0),
                float(k1),
                float(k2),
            )
            calibrations.append(Cal3Bundler(fx, k1, k2, u0, v0))

    assert len(calibrations) == num_cams
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
    # Assumes all camera models have five intrinsic parameters
    camera_model = "RADIAL"

    file_path = os.path.join(save_dir, "cameras.txt")
    with open(file_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        # note that we save the number of estimated cameras, not the number of input images,
        # which would instead be gtsfm_data.number_images().
        f.write(f"# Number of cameras: {len(gtsfm_data.get_valid_camera_indices())}\n")

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
    # ignore first 4 lines of text -- they contain a description of the file format
    # and a record of the number of reconstructed images.
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
    Note: the "Number of images" saved to the .txt file is not the number of images
    fed to the SfM algorithm, but rather the number of localized camera poses/images,
    which COLMAP refers to as the "reconstructed cameras".

    Args:
        gtsfm_data: scene data to write.
        images: list of all images for this scene, in order of image index.
        save_dir: folder to put the images.txt file in.
    """
    os.makedirs(save_dir, exist_ok=True)

    num_imgs = gtsfm_data.number_images()

    image_id_num_measurements = defaultdict(int)
    for j in range(gtsfm_data.number_tracks()):
        track = gtsfm_data.get_track(j)
        for k in range(track.numberMeasurements()):
            image_id, uv_measured = track.measurement(k)
            image_id_num_measurements[image_id] += 1
    mean_obs_per_img = (
        sum(image_id_num_measurements.values()) / len(image_id_num_measurements)
        if len(image_id_num_measurements)
        else 0
    )

    file_path = os.path.join(save_dir, "images.txt")
    with open(file_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {num_imgs}, mean observations per image: {mean_obs_per_img:.3f}\n")

        for i in gtsfm_data.get_valid_camera_indices():
            img_fname = images[i].file_name
            camera = gtsfm_data.get_camera(i)
            # COLMAP exports camera extrinsics (cTw), not the poses (wTc), so must invert
            iTw = camera.pose().inverse()
            iRw_quaternion = iTw.rotation().toQuaternion()
            itw = iTw.translation()
            tx, ty, tz = itw
            qw, qx, qy, qz = iRw_quaternion.w(), iRw_quaternion.x(), iRw_quaternion.y(), iRw_quaternion.z()

            f.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {img_fname}\n")

            # write out points2d
            for j in range(gtsfm_data.number_tracks()):
                track = gtsfm_data.get_track(j)
                for k in range(track.numberMeasurements()):
                    # write each measurement
                    image_id, uv_measured = track.measurement(k)
                    if image_id == i:
                        f.write(f" {uv_measured[0]:.3f} {uv_measured[1]:.3f} {j}")
            f.write("\n")


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


def read_scene(
    images_fpath: str, cameras_fpath: str, points_fpath: str
) -> Tuple[List[Pose3], List[str], List[Cal3Bundler], np.ndarray, np.ndarray]:
    """Reads in full scene reconstruction model."""
    wTi_list, img_fnames = read_images_txt(images_fpath)
    calibrations = read_cameras_txt(cameras_fpath)
    point_cloud, rgb = read_points_txt(points_fpath)
    if any(x is None for x in [wTi_list, img_fnames, calibrations, point_cloud, rgb]):
        raise RuntimeError("One or more of the requested model data products was not found.")
    return wTi_list, img_fnames, calibrations, point_cloud, rgb


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

            for k in range(track.numberMeasurements()):
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


def read_from_bz2_file(file_path: Path) -> Optional[Any]:
    """Reads data using pickle from a compressed file, if it exists."""
    if not file_path.exists():
        return None

    try:
        data = pickle.load(BZ2File(file_path, "rb"))
    except Exception:
        logger.exception("Cache file was corrupted, removing it...")
        os.remove(file_path)
        data = None

    return data


def write_to_bz2_file(data: Any, file_path: Path) -> None:
    """Writes data using pickle to a compressed file."""
    file_path.parent.mkdir(exist_ok=True, parents=True)
    pickle.dump(data, BZ2File(file_path, "wb"))


def save_point_cloud_as_ply(save_fpath: str, points: np.ndarray, rgb: Optional[np.ndarray] = None) -> None:
    """Save a point cloud as a .ply file.

    Args:
        save_fpath: absolute file path where PLY file should be saved.
        points: float array of shape (N,3) representing a 3d point cloud.
        rgb: uint8 array of shape (N,3) representing an RGB color per point.
    """
    if rgb is None:
        # If no colors are provided, then color all points uniformly as black.
        N = points.shape[0]
        rgb = np.zeros((N, 3), dtype=np.uint8)
    pointcloud = open3d_vis_utils.create_colored_point_cloud_open3d(point_cloud=points, rgb=rgb)

    os.makedirs(Path(save_fpath).parent, exist_ok=True)
    open3d.io.write_point_cloud(save_fpath, pointcloud, write_ascii=False, compressed=False, print_progress=False)


def read_point_cloud_from_ply(ply_fpath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a point cloud from a .ply file.

    Args:
        ply_fpath: absolute file path where PLY file is located on disk.

    Returns:
        points: float array of shape (N,3) representing a 3d point cloud.
        rgb: uint8 array of shape (N,3) representing an RGB color per point.
    """
    pointcloud = open3d.io.read_point_cloud(ply_fpath)
    return open3d_vis_utils.convert_colored_open3d_point_cloud_to_numpy(pointcloud)
