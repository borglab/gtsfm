"""Utility to convert GTSAM types to pycolmap types.

Authors: John Lambert
"""
from typing import Dict, Union, Tuple

import pycolmap
import gtsam

import gtsfm.common.types as gtsfm_types
import thirdparty.colmap.scripts.python.read_write_model as colmap


def get_pycolmap_camera(camera_intrinsics: gtsam.Cal3Bundler) -> pycolmap.Camera:
    """Convert Cal3Bundler intrinsics to a pycolmap-compatible format (a dictionary).

    See https://colmap.github.io/cameras.html#camera-models for info about the COLMAP camera models.
    Both SIMPLE_PINHOLE and SIMPLE_RADIAL use 1 focal length.

    Note: the image width and image height values approximated below are dummy placeholder values.
    For some datasets we have intrinsics (including principal point) where image height, image width
    would not necessarily be 2 * cy, 2 * cx. However, image dimensions aren't used anywhere
    in the F / E / H estimation; rather cx and cy are used in the Essential matrix estimation:
    https://github.com/colmap/colmap/blob/9f3a75ae9c72188244f2403eb085e51ecf4397a8/src/base/camera_models.h#L629)

    Args:
        camera_intrinsics: camera intrinsic parameters.
    """
    focal_length = camera_intrinsics.fx()
    cx, cy = camera_intrinsics.px(), camera_intrinsics.py()

    width = int(cx * 2)
    height = int(cy * 2)

    camera_dict = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=width,
        height=height,
        params=[focal_length, cx, cy],
    )
    return camera_dict


def get_calibration_from_colmap(camera: Union[colmap.Camera, pycolmap.Camera]) -> gtsfm_types.CAMERA_TYPE:
    """Associates appropriate GTSfM camera model with the corresponding COLMAP camera model."""
    model_name = camera.model if isinstance(camera, colmap.Camera) else camera.model_name
    if model_name == "SIMPLE_PINHOLE":
        focal_length, cx, cy = camera.params
        return gtsam.Cal3Bundler(focal_length, 0.0, 0.0, cx, cy)
    if model_name == "SIMPLE_RADIAL":
        focal_length, cx, cy, k1 = camera.params
        return gtsam.Cal3Bundler(focal_length, k1, 0.0, cx, cy)
    if model_name == "PINHOLE":
        fx, fy, cx, cy = camera.params
        return gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
    else:
        raise TypeError(f"Unsupported COLMAP camera model {model_name}.")


def point3d_to_sfmtrack(
    point3d: Union[colmap.Point3D, pycolmap.Point3D],
    images: Union[Dict[int, colmap.Image], Dict[int, pycolmap.Image]],
    cameras: Union[Dict[int, colmap.Camera], Dict[int, pycolmap.Camera]],
    invert_pose: bool = False,
) -> Tuple[gtsam.SfmTrack, Dict[int, gtsfm_types.CAMERA_TYPE]]:
    """Convert COLMAP's Point3D object to an SfMTrack.

    Note: the original COLMAP and pyCOLMAP types differ slightly, and since our codebase currently contains methods that
    utilize both implementations, it makes sense to support both.

    Args:
        point3d: COLMAP Point3D containing the landmark as well as lists containing the images from which it was
            observed and the indices of the associated measurement in each image.
        images: COLMAP Image containing camera extrinsics and keypoint measurements.
        cameras: Colmap Camera containing camera intrinsics.
        invert_pose: whether or not to invert the pose contained in the COLMAP Image, which defines the transformation
            from the world frame to the camera frame. This is useful when the user would like to use the returned GTSfM
            types for triangulation.

    Returns:
        The associated SfmTrack and a Dictionary of the cameras from which it was observed.
    """
    if (
        isinstance(point3d, colmap.Point3D)
        and all([isinstance(image, colmap.Image) for image in images.values()])
        and all([isinstance(camera, colmap.Camera) for camera in cameras.values()])
    ):
        track, gtsfm_cameras = colmap_point3d_to_sfmtrack(point3d, images, cameras, invert_pose)
    elif (
        isinstance(point3d, pycolmap.Point3D)
        and all([isinstance(image, pycolmap.Image) for image in images.values()])
        and all([isinstance(camera, pycolmap.Camera) for camera in cameras.values()])
    ):
        track, gtsfm_cameras = pycolmap_point3d_to_sfmtrack(point3d, images, cameras, invert_pose)
    else:
        raise TypeError(
            "Incompatible function arguments. The following argument types are supported:\n"
            + "\t 1. point3d_to_sfmtrack(colmap.Point3D, Dict[int, colmap.Image], Dict[int, colmap.Camera])\n"
            # + "\t 2. point3d_to_sfmtrack(pycolmap.Point3D, Dict[int, pycolmap.Image], Dict[int, pycolmap.Camera])\n"
        )

    return track, gtsfm_cameras


def colmap_point3d_to_sfmtrack(
    point3d: colmap.Point3D,
    images: Dict[int, colmap.Image],
    cameras: Dict[int, colmap.Camera],
    invert_pose: bool = False,
) -> Tuple[gtsam.SfmTrack, Dict[int, gtsfm_types.CAMERA_TYPE]]:
    """Convert COLMAP's Point3D object to an SfMTrack."""
    # TODO (travisdriver): Add RGB values to GTSAM constructor.
    track = gtsam.SfmTrack(point3d.xyz)
    gtsfm_cameras = {}
    for image_id, point2d_idx in zip(point3d.image_ids, point3d.point2D_idxs):
        image = images[image_id]
        track.addMeasurement(image_id, image.xys[point2d_idx])
        calibration = get_calibration_from_colmap(cameras[image.camera_id])
        camera_class = gtsfm_types.get_camera_class_for_calibration(calibration)
        T = gtsam.Pose3(gtsam.Rot3(image.qvec2rotmat()), gtsam.Point3(image.tvec))  # cTw
        if invert_pose:
            T = T.inverse()  # wTc
        gtsfm_cameras[image_id] = camera_class(T, calibration)

    return track, gtsfm_cameras


def pycolmap_point3d_to_sfmtrack(
    point3d: pycolmap.Point3D,
    images: Dict[int, pycolmap.Image],
    cameras: Dict[int, pycolmap.Camera],
    invert_pose: bool = False,
) -> Tuple[gtsam.SfmTrack, Dict[int, gtsfm_types.CAMERA_TYPE]]:
    """Convert pyCOLMAP's Point3D object to an SfMTrack."""
    # TODO (travisdriver): Add RGB values to GTSAM constructor.
    track = gtsam.SfmTrack(point3d.xyz)
    gtsfm_cameras = {}
    for ele in point3d.track.elements:
        image = images[ele.image_id]
        track.addMeasurement(ele.image_id, image.points2D[ele.point2D_idx].xy)
        calibration = get_calibration_from_colmap(cameras[image.camera_id])
        camera_class = gtsfm_types.get_camera_class_for_calibration(calibration)
        T = gtsam.Pose3(gtsam.Rot3(image.rotmat()), gtsam.Point3(image.tvec))  # cTw
        if invert_pose:
            T = T.inverse()  # wTc
        gtsfm_cameras[ele.image_id] = camera_class(T, calibration)

    return track, gtsfm_cameras
