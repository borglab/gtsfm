"""Data structure to hold results of bundle adjustment.

Authors: Xiaolong Wu, Ayush Baid
"""

import numpy as np
from gtsam import SfmData, SfmTrack, Values, symbol_shorthand

C = symbol_shorthand.C
P = symbol_shorthand.P


class SfmResult:
    """Class to hold optimized camera params, 3d landmarks (w/ tracks), and
    total reprojection error.
    """

    def __init__(
        self,
        initial_data: SfmData,
        optimization_result: Values,
        total_reproj_error: float,
    ) -> None:

        self.total_reproj_error = total_reproj_error

        self.result_data = SfmData()

        # add camera params
        for i in range(initial_data.number_cameras()):
            self.result_data.add_camera(
                optimization_result.atPinholeCameraCal3Bundler(C(i))
            )

        # add tracks
        for j in range(initial_data.number_tracks()):
            input_track = initial_data.track(j)

            # init the result with optimized 3D point
            result_track = SfmTrack(optimization_result.atPoint3(P(j)))

            for k in range(input_track.number_measurements()):
                cam_idx, uv = input_track.measurement(k)
                result_track.add_measurement(cam_idx, uv)

            self.result_data.add_track(result_track)

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
        if (
            self.result_data.number_cameras()
            != other.result_data.number_cameras()
        ):
            return False

        # compare camera intrinsics
        for i in range(self.result_data.number_cameras()):
            if (
                not self.result_data.camera(i)
                .calibration()
                .equals(other.result_data.camera(i).calibration(), 1e-1)
            ):
                return False

        # TODO: add pose comparison once the function is in master

        return np.isclose(
            self.total_reproj_error,
            other.total_reproj_error,
            rtol=1e-2,
            atol=1e-1,
        )

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

            camera = self.result_data.camera(cam_idx)

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

        filtered_data = SfmData()

        # add camera params
        for i in range(self.result_data.number_cameras()):
            filtered_data.add_camera(self.result_data.camera(i))

        for j in range(self.result_data.number_tracks()):
            track = self.result_data.track(j)

            if self.__validate_track(track, reproj_err_thresh):
                filtered_data.add_track(track)

        return filtered_data
