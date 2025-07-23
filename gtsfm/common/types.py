"""Common definitions and helper functions for calibration and camera types

Authors: Ayush Baid
"""
from typing import Union

import gtsam


CALIBRATION_TYPE = Union[gtsam.Cal3Bundler, gtsam.Cal3DS2, gtsam.Cal3Fisheye]
CAMERA_TYPE = Union[gtsam.PinholeCameraCal3Bundler, gtsam.PinholeCameraCal3DS2, gtsam.PinholeCameraCal3Fisheye]
CAMERA_SET_TYPE = Union[gtsam.CameraSetCal3Bundler, gtsam.CameraSetCal3DS2, gtsam.CameraSetCal3Fisheye]
PRIOR_FACTOR_TYPE = Union[
    gtsam.PriorFactorCal3Bundler,
    gtsam.PriorFactorCal3DS2,
    gtsam.PriorFactorCal3Fisheye,
]
SFM_FACTOR_TYPE = Union[
    gtsam.GeneralSFMFactor2Cal3Bundler,
    gtsam.GeneralSFMFactor2Cal3DS2,
    gtsam.GeneralSFMFactor2Cal3Fisheye,
]


def get_camera_class_for_calibration(calibration: CALIBRATION_TYPE) -> CAMERA_TYPE:
    """Get the camera class corresponding to the calibration.

    Args:
        calibration: the calibration object for which track is required.

    Returns:
        Camera class needed for the calibration object.
    """
    if isinstance(calibration, gtsam.Cal3Bundler):
        return gtsam.PinholeCameraCal3Bundler
    if isinstance(calibration, gtsam.Cal3DS2):
        return gtsam.PinholeCameraCal3DS2
    if isinstance(calibration, gtsam.Cal3Fisheye):
        return gtsam.PinholeCameraCal3Fisheye
    else:  # If the calibration type is not recognized, raise an error.
        raise ValueError(
            f"Unsupported calibration type: {type(calibration)}. Supported types are {CAMERA_TYPE}."
        )


def get_camera_set_class_for_calibration(calibration: CALIBRATION_TYPE) -> CAMERA_SET_TYPE:
    """Get the camera set class corresponding to the calibration.

    Args:
        calibration: the calibration object for which track is required.

    Returns:
        Camera set class needed for the calibration object.
    """
    if isinstance(calibration, gtsam.Cal3Bundler):
        return gtsam.CameraSetCal3Bundler
    if isinstance(calibration, gtsam.Cal3DS2):
        return gtsam.CameraSetCal3DS2
    if isinstance(calibration, gtsam.Cal3Fisheye):
        return gtsam.CameraSetCal3Fisheye
    else:  # If the calibration type is not recognized, raise an error.
        raise ValueError(
            f"Unsupported calibration type: {type(calibration)}. Supported types are {CAMERA_SET_TYPE}."
        )


def get_prior_factor_for_calibration(calibration: CALIBRATION_TYPE) -> PRIOR_FACTOR_TYPE:
    """Get the prior factor corresponding to the calibration.

    Args:
        calibration: the calibration object for which track is required.

    Returns:
        Prior factor needed for the calibration object.
    """
    if isinstance(calibration, gtsam.Cal3Bundler):
        return gtsam.PriorFactorCal3Bundler
    if isinstance(calibration, gtsam.Cal3DS2):
        return gtsam.PriorFactorCal3DS2
    if isinstance(calibration, gtsam.Cal3Fisheye):
        return gtsam.PriorFactorCal3Fisheye
    else:  # If the calibration type is not recognized, raise an error.
        raise ValueError(
            f"Unsupported calibration type: {type(calibration)}. Supported types are {CALIBRATION_TYPE}."
        )


def get_sfm_factor_for_calibration(calibration: CALIBRATION_TYPE) -> SFM_FACTOR_TYPE:
    """Get the SFM factor corresponding to the calibration.

    Args:
        calibration: the calibration object for which track is required.

    Returns:
        SFM factor needed for the calibration object.
    """
    if isinstance(calibration, gtsam.Cal3Bundler):
        return gtsam.GeneralSFMFactor2Cal3Bundler
    if isinstance(calibration, gtsam.Cal3DS2):
        return gtsam.GeneralSFMFactor2Cal3DS2
    if isinstance(calibration, gtsam.Cal3Fisheye):
        return gtsam.GeneralSFMFactor2Cal3Fisheye
    else:  # If the calibration type is not recognized, raise an error.
        raise ValueError(
            f"Unsupported calibration type: {type(calibration)}. Supported types are {SFM_FACTOR_TYPE}."
        )
