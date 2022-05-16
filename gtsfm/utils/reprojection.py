from typing import Dict, List

import numpy as np
from gtsam import PinholeCameraCal3Bundler, SfmTrack

from gtsfm.common.sfm_track import SfmMeasurement

"""
Note: cannot consolidate the two functions below, since SfmTrack has no measurements() method from C++
"""


def compute_track_reprojection_errors(
    track_cameras: Dict[int, PinholeCameraCal3Bundler], track: SfmTrack
) -> np.ndarray:
    """Compute reprojection errors for measurements in the tracks.

    Args:
        track_cameras: Dict of cameras, with camera indices as keys.
        track: 3d point/landmark and its corresponding 2d measurements in various cameras

    Returns:
        reprojection errors for each measurement (measured in pixels).
    """
    nan = np.array([np.nan, np.nan])
    point = track.point3()
    safe_projections = [track_cameras[i].projectSafe(point) for i, _ in track.measurements]
    valid_2d_errors = np.array(
        [
            uv_projected - uv_measured if success else nan
            for (uv_projected, success), (i, uv_measured) in zip(safe_projections, track.measurements)
        ]
    )
    return np.linalg.norm(valid_2d_errors, axis=1)


def compute_point_reprojection_errors(
    cameras: Dict[int, PinholeCameraCal3Bundler], point3d: np.ndarray, measurements: List[SfmMeasurement]
) -> np.ndarray:
    """Compute reprojection errors for a hypothesized 3d point vs. 2d measurements.

    Args:
        cameras: Dict of cameras, with camera indices as keys.
        point3d: hypothesized 3d point/landmark
        measurements: corresponding 2d measurements (of 3d point above) in various cameras

    Returns:
        reprojection errors for each measurement (measured in pixels).
    """
    nan = np.array([np.nan, np.nan])
    safe_projections = [
        cameras[i].projectSafe(point3d) if i in cameras else (nan, False) for i, uv_measured in measurements
    ]
    valid_2d_errors = np.array(
        [
            uv_projected - uv_measured if success else nan
            for (uv_projected, success), (i, uv_measured) in zip(safe_projections, measurements)
        ]
    )
    return np.linalg.norm(valid_2d_errors, axis=1)
