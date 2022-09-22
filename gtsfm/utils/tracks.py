"""Utilities for 2D and 3D tracks.

Authors: Ayush Baid
"""
from typing import Dict, List

from gtsam import PinholeCameraCal3Bundler, SfmTrack

from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationExitCode,
    TriangulationOptions,
    TriangulationSamplingMode,
)


def classify_tracks2d_with_gt_cameras(
    tracks: List[SfmTrack2d], cameras_gt: List[PinholeCameraCal3Bundler], reproj_error_thresh_px: float = 3
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
    cameras_dict: Dict[int, PinholeCameraCal3Bundler] = {i: cam for i, cam in enumerate(cameras_gt)}
    point3d_initializer = Point3dInitializer(
        track_camera_dict=cameras_dict,
        options=TriangulationOptions(
            reproj_error_threshold=reproj_error_thresh_px, mode=TriangulationSamplingMode.NO_RANSAC
        ),
    )

    exit_codes: List[TriangulationExitCode] = []
    for track in tracks:
        _, _, triangulation_exit_code = point3d_initializer.triangulate(track_2d=track)
        exit_codes.append(triangulation_exit_code)

    return exit_codes


def classify_tracks3d_with_gt_cameras(
    tracks: List[SfmTrack], cameras_gt: List[PinholeCameraCal3Bundler], reproj_error_thresh_px: float = 3
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
