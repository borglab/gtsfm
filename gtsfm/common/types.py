"""Common definitions and helper functions for calibration and camera types

Authors: Ayush Baid
"""
from typing import Union

from gtsam import Cal3Bundler, Cal3Fisheye, PinholeCameraCal3Bundler, PinholeCameraCal3Fisheye

CALIBRATION_TYPE = Union[Cal3Bundler, Cal3Fisheye]
CAMERA_TYPE = Union[PinholeCameraCal3Bundler, PinholeCameraCal3Fisheye]


def get_camera_class_for_calibration(calibration: CALIBRATION_TYPE):
    """Get the camera class corresponding to the calibration.

    Args:
        calibration: the calibration object for which track is required.

    Returns:
        Camera class needed for the calibration object.
    """
    print("Determining camera class for calibration:", calibration, type(calibration))
    if isinstance(calibration, Cal3Bundler):
        print("Using PinholeCameraCal3Bundler for calibration:", calibration)
        return PinholeCameraCal3Bundler
    else:
        print("Using PinholeCameraCal3Fisheye for calibration:", calibration)
        return PinholeCameraCal3Fisheye
