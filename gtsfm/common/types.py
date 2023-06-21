"""Common definitions and helper functions for calibration and camera types

Authors: Ayush Baid
"""
from typing import Union

from gtsam import (
    Cal3Bundler,
    Cal3Fisheye,
    Cal3_S2,
    PinholeCameraCal3Bundler,
    PinholeCameraCal3Fisheye,
    PinholeCameraCal3_S2,
)


CALIBRATION_TYPE = Union[Cal3Bundler, Cal3Fisheye, Cal3_S2]
CAMERA_TYPE = Union[PinholeCameraCal3Bundler, PinholeCameraCal3Fisheye]


def get_camera_class_for_calibration(calibration: CALIBRATION_TYPE):
    """Get the camera class corresponding to the calibration.

    Args:
        calibration: the calibration object for which track is required.

    Returns:
        Camera class needed for the calibration object.
    """
    if isinstance(calibration, Cal3Bundler):
        return PinholeCameraCal3Bundler
    elif isinstance(calibration, Cal3Fisheye):
        return PinholeCameraCal3Fisheye
    elif isinstance(calibration, Cal3_S2):
        return PinholeCameraCal3_S2
    else:
        raise TypeError(f"Unsupported calibration type {type(calibration)}.")
