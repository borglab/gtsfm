"""Data structure to hold results of bundle adjustment.

Authors: Xiaolong Wu, Ayush Baid
"""
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from gtsam import PinholeCameraCal3Bundler, Pose3

import gtsfm.utils.geometry_comparisons as geom_comparisons
from gtsfm.data_association.feature_tracks import SfmTrack


class SfmData:
    """Class to hold cameras, 3D points and their tracks."""

    def __init__(
        self,
        cameras: Optional[Dict[int, PinholeCameraCal3Bundler]] = None,
        tracks: Optional[List[SfmTrack]] = None,
    ) -> None:
        """Initialize from existing cameras and tracks.

        Args:
            cameras (optional): Initial list of cameras. Defaults to None.
            tracks (Optional[List[SfmTrack2dWithLandmark]], optional): [description]. Defaults to None.
        """
        if cameras is not None:
            self.cameras = cameras
        else:
            self.cameras = {}

        if tracks is not None:
            self.tracks = tracks
        else:
            self.tracks = []

    def number_cameras(self) -> int:
        return len(self.cameras)

    def number_tracks(self) -> int:
        return len(self.tracks)

    def add_camera(self, idx: int, camera: PinholeCameraCal3Bundler) -> None:
        self.cameras[idx] = camera

    def add_track(self, track: SfmTrack) -> None:
        self.tracks.append(track)

    def camera(self, idx: int) -> PinholeCameraCal3Bundler:
        return self.cameras[idx]

    def track(self, idx: int) -> SfmTrack:
        return self.tracks[idx]

    def get_track_length_statistics(self) -> Tuple[float, float]:
        """Compute mean and median of track length.

        Returns:
            Mean of track length.
            Median of track length.
        """
        track_lens = map(len, self.tracks)

        return np.mean(track_lens), np.median(track_lens)

    def __validate_track(
        self, track: SfmTrack, reproj_err_thresh: float
    ) -> bool:
        """Validates a track based on reprojection errors and cheirality checks.

        Args:
            track: track with 3D landmark and measurements.
            reproj_err_thresh: reprojection err threshold for each measurement.

        Returns:
            validity of the track.
        """

        for k in range(track.number_measurements()):
            cam_idx, uv = track.measurement(k)

            camera = self.camera(cam_idx)

            # Project to camera
            uv_reprojected, success_flag = camera.projectSafe(track.point3)

            if not success_flag:
                return False

            reproj_error = np.linalg.norm(uv - uv_reprojected)

            if reproj_error > reproj_err_thresh:
                return False

        return True

    def filter_landmarks(self, reproj_err_thresh: float = 5) -> "SfmData":
        """Filters out landmarks with high reprojection error

        Args:
            reproj_err_thresh: reprojection err threshold for each measurement.
        """
        filtered_tracks = [
            track
            for track in self.tracks
            if self.__validate_track(track, reproj_err_thresh)
        ]

        return SfmData(self.cameras, filtered_tracks)


class SfmResult(NamedTuple):
    """Class to hold optimized camera params, 3d landmarks (w/ tracks), and
    total reprojection error.
    """

    sfm_data: SfmData
    total_reproj_error: float

    def __eq__(self, other: object) -> bool:
        """Tests for equality using global poses, intrinsics, and total reprojection error.

        Args:
            other: object to compare.

        Returns:
            Results of equality comparison.
        """
        if not isinstance(other, SfmResult):
            return False

        # compare the number of cameras
        if self.sfm_data.number_cameras() != other.sfm_data.number_cameras():
            return False

        # compare camera intrinsics
        for i in range(self.sfm_data.number_cameras()):
            if (
                not self.sfm_data.camera(i)
                .calibration()
                .equals(other.sfm_data.camera(i).calibration(), 1e-1)
            ):
                return False

        # comparing poses
        poses = self.get_camera_poses()
        other_poses = other.get_camera_poses()

        if not geom_comparisons.compare_global_poses(poses, other_poses):
            return False

        # finally, compare reprojection error
        return np.isclose(
            self.total_reproj_error,
            other.total_reproj_error,
            rtol=1e-2,
            atol=1e-1,
        )

    def get_camera_poses(self) -> List[Pose3]:
        """Getter for camera poses wTi.

        Returns:
            camera poses as a list, each representing wTi
        """
        return [cam.pose() for cam in self.sfm_data.cameras.values()]
