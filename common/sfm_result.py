"""Data structure to hold results of bundle adjustment.

Authors: Xiaolong Wu, Ayush Baid
"""

import numpy as np
from gtsam import SfmData, SfmTrack, VectorValues, symbol_shorthand

C = symbol_shorthand.C
P = symbol_shorthand.P


class SfmResult:
    """Class to hold optimized camera params, 3d landmarks (w/ tracks), and 
    total reprojection error.
    """

    def __init__(self,
                 initial_data: SfmData,
                 optimization_result: VectorValues,
                 total_reproj_error: float) -> None:

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
            result_track = SfmTrack(
                optimization_result.atPoint3(P(j))
            )

            for k in range(input_track.number_measurements()):
                result_track.add_measurement(
                    k,
                    input_track.measurement(k)[1].reshape(2, 1))

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
        if self.result_data.number_cameras() != \
                other.result_data.number_cameras():
            return False

        # compare camera intrinsics
        for i in range(self.result_data.number_cameras()):
            if not self.result_data.camera(i).calibration().equals(
                    other.result_data.camera(i).calibration(), 1e-1):
                return False

        # TODO: add pose comparison once the function is in master

        return np.isclose(
            self.total_reproj_error, other.total_reproj_error,
            rtol=1e-2, atol=1e-1)
