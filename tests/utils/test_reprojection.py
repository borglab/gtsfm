import numpy as np
import unittest

import gtsfm.utils.reprojection as reproj_utils
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3, SfmTrack
from gtsfm.common.sfm_track import SfmMeasurement


class TestReprojection(unittest.TestCase):
    def test_compute_track_reprojection_errors(self):
        """Ensure that reprojection error is computed properly within a track.

        # For camera 0:
        # [13] = [10,0,3]   [1,0,0 | 0]  [1]
        # [24] = [0,10,4] * [0,1,0 | 0] *[2]
        #  [1] = [0, 0,1]   [0,0,1 | 0]  [1]
        #                                [1]

        # For camera 1:
        # [-7] = [10,0,3]   [1,0,0 |-2]  [1]
        # [44] = [0,10,4] * [0,1,0 | 2] *[2]
        #  [1] = [0, 0,1]   [0,0,1 | 0]  [1]
        #                                [1]
        """
        wTi0 = Pose3(Rot3.RzRyRx(0, 0, 0), np.zeros((3, 1)))
        wTi1 = Pose3(Rot3.RzRyRx(0, 0, 0), np.array([2, -2, 0]))

        f = 10
        k1 = 0
        k2 = 0
        u0 = 3
        v0 = 4

        K0 = Cal3Bundler(f, k1, k2, u0, v0)
        K1 = Cal3Bundler(f, k1, k2, u0, v0)

        track_cameras = {0: PinholeCameraCal3Bundler(wTi0, K0), 1: PinholeCameraCal3Bundler(wTi1, K1)}

        triangulated_pt = np.array([1, 2, 1])
        track_3d = SfmTrack(triangulated_pt)

        # in camera 0
        track_3d.addMeasurement(idx=0, m=np.array([13, 24]))
        # in camera 1
        track_3d.addMeasurement(idx=1, m=np.array([-8, 43]))  # should be (-7,44), 1 px error in each dim

        errors = reproj_utils.compute_track_reprojection_errors(track_cameras, track_3d)

        expected_errors = np.array([0, np.sqrt(2)])
        np.testing.assert_allclose(errors, expected_errors)

    def test_compute_point_reprojection_errors(self):
        """Ensure a hypothesized 3d point is projected correctly and compared w/ 2 measurements.
        # For camera 0:
        # [13] = [10,0,3]   [1,0,0 | 0]  [1]
        # [24] = [0,10,4] * [0,1,0 | 0] *[2]
        #  [1] = [0, 0,1]   [0,0,1 | 0]  [1]
        #                                [1]

        # For camera 1:
        # [-7] = [10,0,3]   [1,0,0 |-2]  [1]
        # [44] = [0,10,4] * [0,1,0 | 2] *[2]
        #  [1] = [0, 0,1]   [0,0,1 | 0]  [1]
        #                                [1]
        """
        wTi0 = Pose3(Rot3.RzRyRx(0, 0, 0), np.zeros((3, 1)))
        wTi1 = Pose3(Rot3.RzRyRx(0, 0, 0), np.array([2, -2, 0]))

        f = 10
        k1 = 0
        k2 = 0
        u0 = 3
        v0 = 4

        K0 = Cal3Bundler(f, k1, k2, u0, v0)
        K1 = Cal3Bundler(f, k1, k2, u0, v0)

        track_cameras = {0: PinholeCameraCal3Bundler(wTi0, K0), 1: PinholeCameraCal3Bundler(wTi1, K1)}
        point3d = np.array([1, 2, 1])
        measurements = [
            SfmMeasurement(i=1, uv=np.array([-8, 43])),
            SfmMeasurement(i=2, uv=np.array([0, 0])),  # not in track_cameras!
            SfmMeasurement(i=0, uv=np.array([13, 24])),
        ]

        errors = reproj_utils.compute_point_reprojection_errors(track_cameras, point3d, measurements)
        expected_errors = np.array([np.sqrt(2), np.nan, 0])
        np.testing.assert_allclose(errors, expected_errors)


if __name__ == "__main__":
    unittest.main()
