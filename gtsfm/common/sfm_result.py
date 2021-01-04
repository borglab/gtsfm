"""Data structure to hold results of bundle adjustment.

Authors: Xiaolong Wu, Ayush Baid
"""
from typing import List, NamedTuple, Tuple

import numpy as np
from gtsam import Pose3, SfmData, SfmTrack

import gtsfm.utils.geometry_comparisons as geom_comparisons


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
        return [
            self.sfm_data.camera(i).pose()
            for i in range(self.sfm_data.number_cameras())
        ]

    def get_track_length_statistics(self) -> Tuple[float, float]:
        """Compute mean and median of track length.

        Returns:
            Mean of track length.
            Median of track length.
        """
        track_lens = map(len, self.sfm_data.number_measurements())

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

            camera = self.sfm_data.camera(cam_idx)

            # Project to camera
            uv_reprojected, success_flag = camera.projectSafe(track.point3())

            if not success_flag:
                return False

            reproj_error = np.linalg.norm(uv - uv_reprojected)

            if reproj_error > reproj_err_thresh:
                return False

        return True

    def filter_landmarks(self, reproj_err_thresh: float = 5) -> SfmData:
        """Filters out landmarks with high reprojection error

        Args:
            reproj_err_thresh: reprojection err threshold for each measurement.
        """
        # TODO: move this function to utils or GTSAM
        filtered_data = SfmData()

        # add all the cameras
        for i in range(self.sfm_data.number_cameras()):
            filtered_data.add_camera(self.sfm_data.camera(i))

        for j in range(self.sfm_data.number_tracks()):
            track = self.sfm_data.track(j)

            if self.__validate_track(track, reproj_err_thresh):
                filtered_data.add_track(track)

        return filtered_data
