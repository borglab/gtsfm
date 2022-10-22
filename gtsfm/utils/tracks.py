"""Utilities for 2D and 3D tracks.

Authors: Ayush Baid
"""
import itertools
from typing import Dict, List

import numpy as np
from gtsam import SfmTrack

from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationExitCode,
    TriangulationOptions,
    TriangulationSamplingMode,
)
from gtsfm.common.types import CAMERA_TYPE


def classify_tracks2d_with_gt_cameras(
    tracks: List[SfmTrack2d], cameras_gt: List[CAMERA_TYPE], reproj_error_thresh_px: float = 3
) -> List[TriangulationExitCode]:
    """Classifies the 2D tracks w.r.t ground truth cameras by performing triangulation and collecting exit codes.

    Args:
        tracks: list of 2d tracks.
        cameras_gt: cameras with GT params.
        reproj_error_thresh_px (optional): Reprojection error threshold (in pixels) for a track to be considered an
                                           all-inlier one. Defaults to 3.

    Returns:
        The triangulation exit code for each input track, as list of the same length as of tracks.
    """
    # do a simple triangulation with the GT cameras
    cameras_dict: Dict[int, CAMERA_TYPE] = {i: cam for i, cam in enumerate(cameras_gt)}
    point3d_initializer = Point3dInitializer(
        track_camera_dict=cameras_dict,
        options=TriangulationOptions(
            reproj_error_threshold=reproj_error_thresh_px,
            min_triangulation_angle=-1,
            mode=TriangulationSamplingMode.NO_RANSAC,
        ),
    )

    exit_codes: List[TriangulationExitCode] = []
    for track in tracks:
        _, _, triangulation_exit_code = point3d_initializer.triangulate(track_2d=track)
        exit_codes.append(triangulation_exit_code)

    return exit_codes


def classify_tracks3d_with_gt_cameras(
    tracks: List[SfmTrack], cameras_gt: List[CAMERA_TYPE], reproj_error_thresh_px: float = 3
) -> List[TriangulationExitCode]:
    """Classifies the 3D tracks w.r.t ground truth cameras by performing triangulation and collecting exit codes.

    Args:
        tracks: list of 3d tracks, of length J.
        cameras_gt: cameras with GT params.
        reproj_error_thresh_px (optional): Reprojection error threshold (in pixels) for a track to be considered an
                                           all-inlier one. Defaults to 3.

    Returns:
        The triangulation exit code for each input track, as list of length J (same as input).
    """
    # convert the 3D tracks to 2D tracks
    tracks_2d: List[SfmTrack2d] = []
    for track_3d in tracks:
        num_measurements = track_3d.numberMeasurements()

        measurements: List[SfmMeasurement] = []
        for k in range(num_measurements):
            i, uv = track_3d.measurement(k)

            measurements.append(SfmMeasurement(i, uv))

        tracks_2d.append(SfmTrack2d(measurements))

    return classify_tracks2d_with_gt_cameras(tracks_2d, cameras_gt, reproj_error_thresh_px)


def get_triangulation_angle(track3d: SfmTrack, cameras: Dict[int, CAMERA_TYPE]) -> float:
    """Get the angle subtended by the cameras at the 3D landmark of the track.

    Args:
        track3d: the track with the landmark.
        cameras: cameras which have been used to triangulate the landmark.

    Returns:
        float: the triangulation angle (max over all pairs of cameras associated with the track).
    """
    landmark: np.ndarray[np.float64[3, 1]] = track3d.point3()

    camera_idxs: List[int] = []
    for k in range(track3d.numberMeasurements()):
        i, _ = track3d.measurement(k)
        camera_idxs.append(i)

    angles: List[float] = []
    for i1, i2 in itertools.combinations(camera_idxs, 2):
        camera_i1_center: np.ndarray[np.float64[3, 1]] = cameras[i1].pose().translation()
        camera_i2_center: np.ndarray[np.float64[3, 1]] = cameras[i2].pose().translation()

        camera_i1_ray: np.ndarray[np.float64[3, 1]] = landmark - camera_i1_center
        camera_i2_ray: np.ndarray[np.float64[3, 1]] = landmark - camera_i2_center

        subtended_angle: float = np.rad2deg(
            np.arccos(
                np.dot(camera_i1_ray, camera_i2_ray) / (np.linalg.norm(camera_i1_ray) * np.linalg.norm(camera_i2_ray))
            )
        )

        angles.append(subtended_angle)

    return max(angles)
