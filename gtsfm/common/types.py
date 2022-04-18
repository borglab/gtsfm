from typing import Union

from gtsam import Cal3Bundler, Cal3Fisheye, PinholeCameraCal3Bundler, PinholeCameraCal3Fisheye

CALIBRATION_TYPE = Union[Cal3Bundler, Cal3Fisheye]
CAMERA_TYPE = Union[PinholeCameraCal3Bundler, PinholeCameraCal3Fisheye]


def get_camera_class_for_calibration(calibration: CALIBRATION_TYPE):
    if isinstance(calibration, Cal3Bundler):
        return PinholeCameraCal3Bundler
    else:
        return PinholeCameraCal3Fisheye
