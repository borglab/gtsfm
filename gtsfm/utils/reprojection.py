"""
Module to compute reprojection errors for SfmTracks and 3D points.

Author: John Lambert
"""

from typing import Dict, Iterator, List, Tuple

import numpy as np
from gtsam import SfmTrack

from gtsfm.common.sfm_track import SfmMeasurement
from gtsfm.common.types import CAMERA_TYPE


def _compute_reprojection_errors_core(
    track_camera_dict: Dict[int, CAMERA_TYPE], point3d: np.ndarray, measurements: Iterator[Tuple[int, np.ndarray]]
) -> Tuple[np.ndarray, float]:
    """Core logic for computing reprojection errors using batch norm computation.

    Args:
        track_camera_dict: Dict of cameras, with camera indices as keys.
        point3d: 3D point to project
        measurements: Iterator yielding (camera_idx, uv_measured) tuples

    Returns:
        reprojection errors for each measurement (measured in pixels).
        avg_track_reproj_error: average reprojection error of all measurements.
    """
    uv_measured_list = []
    uv_reprojected_list = []
    valid_mask = []

    for i, uv_measured in measurements:
        if i not in track_camera_dict:
            # camera pose was not successfully estimated, camera was uninitialized
            uv_measured_list.append(uv_measured)
            uv_reprojected_list.append(np.array([np.nan, np.nan]))
            valid_mask.append(False)
            continue

        camera = track_camera_dict[i]
        uv_reprojected, success_flag = camera.projectSafe(point3d)

        uv_measured_list.append(uv_measured)
        uv_reprojected_list.append(uv_reprojected if success_flag else np.array([np.nan, np.nan]))
        valid_mask.append(success_flag)

    # Compute all norms in batch
    uv_measured_array = np.array(uv_measured_list)
    uv_reprojected_array = np.array(uv_reprojected_list)
    errors = np.linalg.norm(uv_measured_array - uv_reprojected_array, axis=1)

    # Set invalid projections to NaN
    errors[~np.array(valid_mask)] = np.nan

    avg_error = np.nan if np.isnan(errors).all() else float(np.nanmean(errors))
    return errors, avg_error


def compute_reprojection_errors_with_stats(
    track_camera_dict: Dict[int, CAMERA_TYPE], point3d: np.ndarray, measurements: Iterator[Tuple[int, np.ndarray]]
) -> Tuple[np.ndarray, float, float, float]:
    """Compute reprojection errors and summary stats for a 3D point and its measurements.

    Args:
        track_camera_dict: Dict of cameras, with camera indices as keys.
        point3d: 3D point to project.
        measurements: Iterator yielding (camera_idx, uv_measured) tuples.

    Returns:
        Tuple of (errors array, mean error, min error, max error). Stats ignore NaNs.
    """
    errors, avg_error = _compute_reprojection_errors_core(track_camera_dict, point3d, measurements)
    if np.isnan(errors).all():
        min_error = max_error = median_error = np.nan
    else:
        min_error = float(np.nanmin(errors))
        max_error = float(np.nanmax(errors))
        median_error = float(np.nanmedian(errors))
    return errors, avg_error, median_error, min_error, max_error


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
    measurements = (track.measurement(k) for k in range(track.numberMeasurements()))
    return _compute_reprojection_errors_core(track_camera_dict, track.point3(), measurements)


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
    return _compute_reprojection_errors_core(track_camera_dict, point3d, iter(measurements))
