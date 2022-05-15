from typing import Dict, List, Tuple

import numpy as np
from gtsam import PinholeCameraCal3Bundler, SfmTrack

from gtsfm.common.sfm_track import SfmMeasurement

"""
Note: cannot consolidate the two functions below, since SfmTrack has no measurements() method from C++
"""


def compute_track_reprojection_errors(
    track_camera_dict: Dict[int, PinholeCameraCal3Bundler], track: SfmTrack
) -> np.ndarray:
    """Compute reprojection errors for measurements in the tracks.

    Args:
        track_camera_dict: Dict of cameras, with camera indices as keys.
        track: 3d point/landmark and its corresponding 2d measurements in various cameras

    Returns:
        reprojection errors for each measurement (measured in pixels).
        average_reprojection_error: average reprojection error of all meausurements in track.
    """
    errors = []
    for k in range(track.numberMeasurements()):

        # process each measurement
        i, uv_measured = track.measurement(k)

        # get the camera associated with the measurement
        camera = track_camera_dict[i]
        # Project to camera
        uv_reprojected, success_flag = camera.projectSafe(track.point3())
        # Projection error in camera
        if success_flag:
            errors.append(np.linalg.norm(uv_measured - uv_reprojected))
        else:
            # failure in projection
            errors.append(np.nan)

    return np.array(errors)


def compute_point_reprojection_errors(
    cameras: Dict[int, PinholeCameraCal3Bundler], point3d: np.ndarray, measurements: List[SfmMeasurement]
) -> Tuple[np.ndarray, float]:
    """Compute reprojection errors for a hypothesized 3d point vs. 2d measurements.

    Args:
        cameras: Dict of cameras, with camera indices as keys.
        point3d: hypothesized 3d point/landmark
        measurements: corresponding 2d measurements (of 3d point above) in various cameras

    Returns:
        reprojection errors for each measurement (measured in pixels).
        average_reprojection_error: average reprojection error of all measurements in track.
    """
    nan = np.array([np.nan, np.nan])
    safe_projections = [
        cameras[i].projectSafe(point3d) if i in cameras else (nan, False) for i, uv_measured in measurements
    ]
    valid_projections = np.array(
        [
            uv_projected - uv_measured
            for (uv_projected, success), (i, uv_measured) in zip(safe_projections, measurements)
        ]
    )
    errors = np.linalg.norm(valid_projections, axis=1)
    average_reprojection_error = np.nan if np.isnan(errors).all() else np.nanmean(errors)
    return errors, average_reprojection_error
