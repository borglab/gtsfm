"""Utility to convert GTSAM types to pycolmap types.

Authors: John Lambert
"""

import pycolmap
from gtsam import Cal3Bundler


def get_pycolmap_camera(camera_intrinsics: Cal3Bundler) -> pycolmap.Camera:
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
