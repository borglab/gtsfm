"""Unit tests for track utils.

Author: Ayush Baid
"""

import unittest

import numpy as np
from gtsam import PinholeCameraCal3Bundler, Pose3, Rot3, SfmTrack

import gtsfm.utils.tracks as track_utils

DUMMY_KEYPOINT = np.array([0, 0], dtype=np.float64)


class TestTrackUtils(unittest.TestCase):
    """Unit tests for track utility functions."""

    def test_get_max_triangulation_angle_with_2_measurements(self) -> None:
        """Test the angle computation with just two measurements."""

        camera_center_i1 = np.array([0, 0, 0], dtype=np.float64)
        camera_center_i2 = np.array([2, 0, 0], dtype=np.float64)
        landmark = np.array([1, 1, 0], dtype=np.float64)
        expected = 90  # in degrees

        i1 = 5
        i2 = 7

        cameras = {}

        cameras[i1] = PinholeCameraCal3Bundler(Pose3(Rot3(), camera_center_i1))
        cameras[i2] = PinholeCameraCal3Bundler(Pose3(Rot3(), camera_center_i2))

        # create dummy measurement
        track = SfmTrack(landmark)
        track.addMeasurement(i1, DUMMY_KEYPOINT)
        track.addMeasurement(i2, DUMMY_KEYPOINT)

        computed = track_utils.get_max_triangulation_angle(track3d=track, cameras=cameras)
        self.assertAlmostEqual(computed, expected)

    def test_get_max_triangulation_angle_with_3_measurements(self) -> None:
        """Test the angle computation with just two measurements."""

        camera_center_i1 = np.array([0, 0, 0], dtype=np.float64)
        camera_center_i2 = np.array([3, 0, 0], dtype=np.float64)
        camera_center_i3 = np.array([1, 0, 0], dtype=np.float64)
        landmark = np.array([1, 1, 0], dtype=np.float64)
        expected = 108.43494882292202  # in degrees

        i1 = 5
        i2 = 7
        i3 = 2

        cameras = {}

        cameras[i1] = PinholeCameraCal3Bundler(Pose3(Rot3(), camera_center_i1))
        cameras[i2] = PinholeCameraCal3Bundler(Pose3(Rot3(), camera_center_i2))
        cameras[i3] = PinholeCameraCal3Bundler(Pose3(Rot3(), camera_center_i3))

        # create dummy measurement
        track = SfmTrack(landmark)
        track.addMeasurement(i1, DUMMY_KEYPOINT)
        track.addMeasurement(i2, DUMMY_KEYPOINT)
        track.addMeasurement(i3, DUMMY_KEYPOINT)

        computed = track_utils.get_max_triangulation_angle(track3d=track, cameras=cameras)
        self.assertAlmostEqual(computed, expected)


if __name__ == "__main__":
    unittest.main()
