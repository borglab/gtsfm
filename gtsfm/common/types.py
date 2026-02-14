"""Common definitions and helper functions for calibration and camera types

Authors: Ayush Baid, Travis Driver
"""

from typing import Type, Union

import gtsam  # type: ignore
import numpy as np

CALIBRATION_TYPE = Union[gtsam.Cal3Bundler, gtsam.Cal3_S2, gtsam.Cal3DS2, gtsam.Cal3Fisheye]
CAMERA_TYPE = Union[
    gtsam.PinholeCameraCal3Bundler,
    gtsam.PinholeCameraCal3_S2,
    gtsam.PinholeCameraCal3DS2,
    gtsam.PinholeCameraCal3Fisheye,
]
CAMERA_SET_TYPE = Union[
    gtsam.CameraSetCal3Bundler, gtsam.CameraSetCal3_S2, gtsam.CameraSetCal3DS2, gtsam.CameraSetCal3Fisheye
]
PRIOR_FACTOR_TYPE = Union[
    gtsam.PriorFactorCal3Bundler,
    gtsam.PriorFactorCal3_S2,
    gtsam.PriorFactorCal3DS2,
    gtsam.PriorFactorCal3Fisheye,
]
SFM_FACTOR_TYPE = Union[
    gtsam.GeneralSFMFactor2Cal3Bundler,
    gtsam.GeneralSFMFactor2Cal3_S2,
    gtsam.GeneralSFMFactor2Cal3DS2,
    gtsam.GeneralSFMFactor2Cal3Fisheye,
]


def get_camera_class_for_calibration(calibration: CALIBRATION_TYPE) -> Type[CAMERA_TYPE]:
    """Get the camera class corresponding to the calibration.

    Args:
        calibration: the calibration object for which track is required.

    Returns:
        Camera class needed for the calibration object.
    """
    if isinstance(calibration, gtsam.Cal3Bundler):
        return gtsam.PinholeCameraCal3Bundler
    if isinstance(calibration, gtsam.Cal3_S2):
        return gtsam.PinholeCameraCal3_S2
    if isinstance(calibration, gtsam.Cal3DS2):
        return gtsam.PinholeCameraCal3DS2
    if isinstance(calibration, gtsam.Cal3Fisheye):
        return gtsam.PinholeCameraCal3Fisheye
    else:  # If the calibration type is not recognized, raise an error.
        raise ValueError(f"Unsupported calibration type: {type(calibration)}. Supported types are {CALIBRATION_TYPE}.")


def create_camera(pose: gtsam.Pose3, calibration: CALIBRATION_TYPE) -> CAMERA_TYPE:
    """Create a camera object given pose and calibration.

    Args:
        pose: the pose of the camera.
        calibration: the calibration object for which track is required.

    Returns:
        A camera object corresponding to the given pose and calibration.
    """
    camera_class = get_camera_class_for_calibration(calibration)
    return camera_class(pose, calibration)  # type: ignore


def get_camera_set_class_for_calibration(calibration: CALIBRATION_TYPE) -> Type[CAMERA_SET_TYPE]:
    """Get the camera set class corresponding to the calibration.

    Args:
        calibration: the calibration object for which track is required.

    Returns:
        Camera set class needed for the calibration object.
    """
    if isinstance(calibration, gtsam.Cal3Bundler):
        return gtsam.CameraSetCal3Bundler
    if isinstance(calibration, gtsam.Cal3_S2):
        return gtsam.CameraSetCal3_S2
    if isinstance(calibration, gtsam.Cal3DS2):
        return gtsam.CameraSetCal3DS2
    if isinstance(calibration, gtsam.Cal3Fisheye):
        return gtsam.CameraSetCal3Fisheye
    else:  # If the calibration type is not recognized, raise an error.
        raise ValueError(f"Unsupported calibration type: {type(calibration)}. Supported types are {CALIBRATION_TYPE}.")


def get_prior_factor_for_calibration(calibration: CALIBRATION_TYPE) -> Type[PRIOR_FACTOR_TYPE]:
    """Get the prior factor corresponding to the calibration.

    Args:
        calibration: the calibration object for which track is required.

    Returns:
        Prior factor needed for the calibration object.
    """
    if isinstance(calibration, gtsam.Cal3Bundler):
        return gtsam.PriorFactorCal3Bundler
    if isinstance(calibration, gtsam.Cal3_S2):
        return gtsam.PriorFactorCal3_S2
    if isinstance(calibration, gtsam.Cal3DS2):
        return gtsam.PriorFactorCal3DS2
    if isinstance(calibration, gtsam.Cal3Fisheye):
        return gtsam.PriorFactorCal3Fisheye
    else:  # If the calibration type is not recognized, raise an error.
        raise ValueError(f"Unsupported calibration type: {type(calibration)}. Supported types are {CALIBRATION_TYPE}.")


def get_noise_model_for_calibration(
    calibration, focal_sigma: float, pp_sigma: float, skew_sigma: float = 1e-6, dist_sigma: float = 1e-6
) -> gtsam.noiseModel.Diagonal:
    """Get the noise model for the calibration, based on the calibration type.

    Convenience function to only set the focal length and principal point noise, and leave the rest at 1e-5.

    Args:
        calibration: the calibration object to get the noise model for.
        focal_sigma: the sigma for the focal length.
        pp_sigma: the sigma for the principal point.

    Returns:
        A Diagonal noise model with the given sigma for the focal length and principal point.
    """
    if isinstance(calibration, gtsam.Cal3Bundler):
        sigmas = np.array([focal_sigma, dist_sigma, dist_sigma])  # f, k1, k2
    elif isinstance(calibration, gtsam.Cal3_S2):
        sigmas = np.array([focal_sigma, focal_sigma, skew_sigma, pp_sigma, pp_sigma])  # fx, fy, s, cx, cy
    elif isinstance(calibration, gtsam.Cal3DS2):
        sigmas = np.array(
            [
                focal_sigma,
                focal_sigma,
                skew_sigma,  # skew
                pp_sigma,
                pp_sigma,
                dist_sigma,  # k1
                dist_sigma,  # k2
                dist_sigma,  # p1
                dist_sigma,  # p2
            ]
        )
    elif isinstance(calibration, gtsam.Cal3Fisheye):
        sigmas = np.array(
            [
                focal_sigma,
                focal_sigma,
                skew_sigma,  # skew
                pp_sigma,
                pp_sigma,
                dist_sigma,  # k1
                dist_sigma,  # k2
                dist_sigma,  # p1
                dist_sigma,  # p2
            ]
        )
    else:  # If the calibration type is not recognized, raise an error.
        raise ValueError(f"Unsupported calibration type: {type(calibration)}. Supported types are {CALIBRATION_TYPE}.")
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


def get_sfm_factor_for_calibration(calibration: CALIBRATION_TYPE) -> Type[SFM_FACTOR_TYPE]:
    """Get the SFM factor corresponding to the calibration.

    Args:
        calibration: the calibration object for which track is required.

    Returns:
        SFM factor needed for the calibration object.
    """
    if isinstance(calibration, gtsam.Cal3Bundler):
        return gtsam.GeneralSFMFactor2Cal3Bundler
    if isinstance(calibration, gtsam.Cal3_S2):
        return gtsam.GeneralSFMFactor2Cal3_S2
    if isinstance(calibration, gtsam.Cal3DS2):
        return gtsam.GeneralSFMFactor2Cal3DS2
    if isinstance(calibration, gtsam.Cal3Fisheye):
        return gtsam.GeneralSFMFactor2Cal3Fisheye
    else:  # If the calibration type is not recognized, raise an error.
        raise ValueError(f"Unsupported calibration type: {type(calibration)}. Supported types are {CALIBRATION_TYPE}.")
