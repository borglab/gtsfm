
import numpy as np

from gtsam import SfmTrack

"""
Note: cannot consolidate the two functions below, since SfmTrack has no measurements() method from C++
"""

def compute_track_reprojection_errors(
    track_camera_dict: Dict[int, PinholeCameraCal3Bundler], track:
) -> Tuple[np.ndarray,float]:
    """Compute reprojection errors for measurements in the tracks.

    Args:
        track_camera_dict: Dict of cameras and their indices.
        track: 3d point/landmark and its corresponding 2d measurements in various cameras

    Returns:
        reprojection errors for each measurement.
        avg_track_reproj_error: average reprojection error of all meausurements in track.
    """
    errors = []
    for i, uv_measured in meausurements:

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

    errors = np.array(errors)
    avg_track_reproj_error = errors.mean()
    return errors, avg_track_reproj_error


def compute_point_reprojection_errors(
    track_camera_dict: Dict[int, PinholeCameraCal3Bundler], point3d: np.ndarray, measurements: List[SfmMeasurement]
) -> Tuple[np.ndarray,float]:
    """Compute reprojection errors for measurements in the tracks.

    Args:
        track_camera_dict: Dict of cameras and their indices.
        point3d: hypothesized 3d point/landmark 
        measurements: corresponding 2d measurements (of 3d point above) in various cameras

    Returns:
        reprojection errors for each measurement.
        avg_track_reproj_error: average reprojection error of all meausurements in track.
    """
    errors = []
    for (i, uv_measured) in meausurements:

         # get the camera associated with the measurement
        camera = track_camera_dict[i]
        # Project to camera
        uv_reprojected, success_flag = camera.projectSafe(point3d)
        # Projection error in camera
        if success_flag:
            errors.append(np.linalg.norm(uv_measured - uv_reprojected))
        else:
            # failure in projection
            errors.append(np.nan)

    errors = np.array(errors)
    avg_track_reproj_error = errors.mean()
    return errors, avg_track_reproj_error

