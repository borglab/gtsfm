"""Utilities for converting between GTSAM to pycolmap types.

Authors: John Lambert
"""

import gtsam  # type: ignore
import pycolmap

from gtsfm.common.types import CALIBRATION_TYPE
from thirdparty.colmap.scripts.python.read_write_model import Camera as ColmapCamera


def get_pycolmap_camera(camera_intrinsics: CALIBRATION_TYPE) -> pycolmap.Camera:
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


def colmap_camera_to_gtsam_calibration(camera: ColmapCamera) -> CALIBRATION_TYPE:
    """Convert a pycolmap camera to a GTSAM Cal3 object in CALIBRATION_TYPE union

    Args:
        camera: A pycolmap camera object.

    Returns:
        A GTSAM Calibration object.
    """
    # TODO(travisdriver): use pycolmap cameras.
    camera_model_name = camera.model

    # Default to zero-valued radial distortion coefficients (quadratic and quartic).
    if camera_model_name == "SIMPLE_RADIAL":
        # See https://github.com/colmap/colmap/blob/1f6812e333a1e4b2ef56aa74e2c3873e4e3a40cd/src/colmap/sensor/models.h#L212  # noqa: E501
        f, cx, cy, k1 = camera.params
        return gtsam.Cal3Bundler(f, k1, 0.0, cx, cy)
    elif camera_model_name == "FULL_OPENCV":
        # See https://github.com/colmap/colmap/blob/1f6812e333a1e4b2ef56aa74e2c3873e4e3a40cd/src/colmap/sensor/models.h#L273  # noqa: E501
        fx, fy, cx, cy, k1, k2, p1, p2 = camera.params[:8]
        return gtsam.Cal3DS2(fx, fy, 0.0, cx, cy, k1, k2, p1, p2)
    elif camera_model_name == "PINHOLE":
        # See https://github.com/colmap/colmap/blob/1f6812e333a1e4b2ef56aa74e2c3873e4e3a40cd/src/colmap/sensor/models.h#L196  # noqa: E501
        fx, fy, cx, cy = camera.params
        return gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
    elif camera_model_name == "RADIAL":
        # See https://github.com/colmap/colmap/blob/1f6812e333a1e4b2ef56aa74e2c3873e4e3a40cd/src/colmap/sensor/models.h#L227  # noqa: E501
        f, cx, cy, k1, k2 = camera.params
        return gtsam.Cal3Bundler(f, k1, k2, cx, cy)
    elif camera_model_name == "OPENCV":
        # See https://github.com/colmap/colmap/blob/1f6812e333a1e4b2ef56aa74e2c3873e4e3a40cd/src/colmap/sensor/models.h#L241  # noqa: E501
        fx, fy, cx, cy, k1, k2, p1, p2 = camera.params
        return gtsam.Cal3DS2(fx, fy, 0.0, cx, cy, k1, k2, p1, p2)
    else:
        raise ValueError(f"Unsupported COLMAP camera type: {camera_model_name}")


def gtsfm_calibration_to_colmap_camera(camera_id, calibration: gtsam.Cal3, height: int, width: int) -> ColmapCamera:
    """Convert a GTSAM calibration object to a pycolmap camera.

    Args:
        calibration: A GTSAM Calibration object.

    Returns:
        A pycolmap camera object.
    """
    if isinstance(calibration, gtsam.Cal3Bundler):
        return ColmapCamera(
            model="RADIAL",
            id=camera_id,
            width=width,
            height=height,
            params=[calibration.fx(), calibration.px(), calibration.py(), calibration.k1(), calibration.k2()],
        )
    elif isinstance(calibration, gtsam.Cal3_S2):
        return ColmapCamera(
            model="PINHOLE",
            id=camera_id,
            width=width,
            height=height,
            params=[calibration.fx(), calibration.fy(), calibration.px(), calibration.py()],
        )
    elif isinstance(calibration, gtsam.Cal3DS2):
        return ColmapCamera(
            model="OPENCV",
            id=camera_id,
            width=width,
            height=height,
            params=[
                calibration.fx(),
                calibration.fy(),
                calibration.px(),
                calibration.py(),
                calibration.k1(),
                calibration.k2(),
                0.0,
                0.0,
                # calibration.p1(),
                # calibration.p2(),
            ],
        )
    else:
        raise ValueError(f"Unsupported calibration type: {type(calibration)}")
