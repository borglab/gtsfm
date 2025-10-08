"""
Module to compute reprojection errors for SfmTracks and 3D points.

Author: John Lambert
"""

from typing import Dict, List, Tuple

import numpy as np
from gtsam import SfmTrack

from gtsfm.common.sfm_track import SfmMeasurement
from gtsfm.common.types import CAMERA_TYPE

"""
Note: cannot consolidate the two functions below, since SfmTrack has no measurements() method from C++
"""


def compute_track_reprojection_errors(
    track_camera_dict: Dict[int, CAMERA_TYPE], track: SfmTrack
) -> Tuple[np.ndarray, float]:
    """Compute reprojection errors for measurements in the tracks.

    Args:
        track_camera_dict: Dict of cameras, with camera indices as keys.
        track: 3d point/landmark and its corresponding 2d measurements in various cameras

    Returns:
        reprojection errors for each measurement (measured in pixels).
        avg_track_reproj_error: average reprojection error of all measurements in track.
    """
    errors: List[float] = []
    for k in range(track.numberMeasurements()):

        # process each measurement
        i, uv_measured = track.measurement(k)

        # get the camera associated with the measurement
        camera = track_camera_dict[i]
        # Project to camera
        uv_reprojected, success_flag = camera.projectSafe(track.point3())
        # Projection error in camera
        if success_flag:
            errors.append(float(np.linalg.norm(uv_measured - uv_reprojected)))
        else:
            # failure in projection
            errors.append(np.nan)

    errors_array = np.array(errors)
    avg_track_reproj_error = np.nan if np.isnan(errors_array).all() else float(np.nanmean(errors_array))
    return errors_array, avg_track_reproj_error


def compute_point_reprojection_errors(
    track_camera_dict: Dict[int, CAMERA_TYPE], point3d: np.ndarray, measurements: List[SfmMeasurement]
) -> Tuple[np.ndarray, float]:
    """Compute reprojection errors for a hypothesized 3d point vs. 2d measurements.

    Args:
        track_camera_dict: Dict of cameras, with camera indices as keys.
        point3d: hypothesized 3d point/landmark
        measurements: corresponding 2d measurements (of 3d point above) in various cameras

    Returns:
        reprojection errors for each measurement (measured in pixels).
        avg_track_reproj_error: average reprojection error of all measurements in track.
    """
    errors = []
    for i, uv_measured in measurements:

        if i not in track_camera_dict:
            # camera pose was not successfully estimated, camera was uninitialized
            errors.append(np.nan)
            continue

        # get the camera associated with the measurement
        camera = track_camera_dict[i]
        # Project to camera
        uv_reprojected, success_flag = camera.projectSafe(point3d)
        # Projection error in camera
        if success_flag:
            errors.append(float(np.linalg.norm(uv_measured - uv_reprojected)))
        else:
            # failure in projection
            errors.append(np.nan)

    errors_array = np.array(errors)
    avg_track_reproj_error = np.nan if np.isnan(errors_array).all() else float(np.nanmean(errors_array))
    return errors_array, avg_track_reproj_error
