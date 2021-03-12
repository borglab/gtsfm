"""Data structure to hold results of bundle adjustment.

Authors: Xiaolong Wu, Ayush Baid
"""
from typing import List, NamedTuple, Tuple

import numpy as np
from gtsam import Pose3, SfmTrack

from gtsfm.common.gtsfm_data import GtsfmData


class SfmResult(NamedTuple):
    """Class to hold optimized camera params, 3d landmarks (w/ tracks), and total reprojection error."""

    gtsfm_data: GtsfmData
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

        if self.gtsfm_data != other.gtsfm_data:
            return False

        # finally, compare reprojection error
        return np.isclose(self.total_reproj_error, other.total_reproj_error, rtol=1e-2, atol=1e-1, equal_nan=True)

    def get_camera_poses(self) -> List[Pose3]:
        """Getter for camera poses wTi.

        Returns:
            camera poses as a list, each representing wTi
        """
        cameras = [self.gtsfm_data.get_camera(i) for i in range(self.gtsfm_data.number_images())]
        poses = [camera.pose() if camera is not None else None for camera in cameras]

        return poses

    def get_track_length_statistics(self) -> Tuple[float, float, np.ndarray]:
        """Compute mean and median lengths of all the tracks.

        Returns:
            Mean track length.
            Median track length.
            Array containing all track lengths.
        """
        if self.gtsfm_data.number_tracks() == 0:
            return 0, 0, np.array([], dtype=np.uint32)

        track_lengths = [
            self.gtsfm_data.get_track(j).number_measurements() for j in range(self.gtsfm_data.number_tracks())
        ]

        return np.mean(track_lengths), np.median(track_lengths), np.array(track_lengths, dtype=np.uint32)

    def __validate_track(self, track: SfmTrack, reproj_err_thresh: float) -> bool:
        """Validates a track based on reprojection errors and cheirality checks.

        Args:
            track: track with 3D landmark and measurements.
            reproj_err_thresh: reprojection err threshold for each measurement.

        Returns:
            validity of the track.
        """

        for k in range(track.number_measurements()):
            # process each measurement
            cam_idx, uv = track.measurement(k)

            # get the camera associated with the measurement
            camera = self.gtsfm_data.get_camera(cam_idx)

            # Project to camera
            uv_reprojected, success_flag = camera.projectSafe(track.point3())

            if not success_flag:
                # failure in projection
                return False

            # compute and check reprojection error
            reproj_error = np.linalg.norm(uv - uv_reprojected)
            if reproj_error > reproj_err_thresh:
                return False

        # track is valid as all measurements have error below the threshold
        return True

    def filter_landmarks(self, reproj_err_thresh: float = 5) -> GtsfmData:
        """Filters out landmarks with high reprojection error

        Args:
            reproj_err_thresh: reprojection err threshold for each measurement.
        """
        # TODO: move this function to utils or GTSAM
        filtered_data = GtsfmData(self.gtsfm_data.number_images())

        # add all the cameras
        for i in self.gtsfm_data.get_valid_camera_indices():
            filtered_data.add_camera(self.gtsfm_data.get_camera(i), i)

        for j in range(self.gtsfm_data.number_tracks()):
            track = self.gtsfm_data.get_track(j)

            if self.__validate_track(track, reproj_err_thresh):
                filtered_data.add_track(track)

        return filtered_data
