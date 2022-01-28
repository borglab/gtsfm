"""Utility to convert GTSAM types to pycolmap types.

Authors: John Lambert
"""

from typing import Any, Dict

import pycolmap
from gtsam import Cal3Bundler


def get_pycolmap_camera(camera_intrinsics: Cal3Bundler) -> pycolmap.Camera:
    """Convert Cal3Bundler intrinsics to a pycolmap-compatible format (a dictionary).

    See https://colmap.github.io/cameras.html#camera-models for info about the COLMAP camera models.
    Both SIMPLE_PINHOLE and SIMPLE_RADIAL use 1 focal length.
    """
    focal_length = camera_intrinsics.fx()
    cx, cy = camera_intrinsics.px(), camera_intrinsics.py()

    # TODO (johnwlambert): use more accurate proxy?
    width = int(cx * 2)
    height = int(cy * 2)

    camera_dict = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=width,
        height=height,
        params=[focal_length, cx, cy],
    )
    return camera_dict
