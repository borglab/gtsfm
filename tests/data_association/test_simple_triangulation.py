"""Unit test for simple (no ransac) based triangulation.

Authors: Ayush Baid
"""
import unittest
from typing import Dict, List

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler

from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.data_association.point3d_initializer import Point3dInitializer, TriangulationParam

# make a pentahedron with 5 vertices, with first 4 vertices describing the base of the pentahedron.
LANDMARK_POINTS_3D = [
    np.array([-10, 20, 50]),
    np.array([-10, 10, 50]),
    np.array([10, 20, 50]),
    np.array([10, 10, 50]),
    np.array([0, 15, 30]),
]

# defining a default calibration
DEFAULT_CALIB = Cal3Bundler(50, 0, 0, 0, 0)

# define a pair of cameras which work well with triangulation (large baseline and pointing towards each other)
CAMERA_PAIR_GOOD = {
    0: PinholeCameraCal3Bundler.Lookat(np.array([0, 0, 0]), np.array([0, 12, 20]), np.array([0, 0, 1]), DEFAULT_CALIB),
    1: PinholeCameraCal3Bundler.Lookat(
        np.array([30, 20, -5]), np.array([0, 12, 20]), np.array([0, 0, 1]), DEFAULT_CALIB
    ),
}

# camera pair with small baseline
CAMERA_PAIR_SMALL_BASELINE = {
    0: PinholeCameraCal3Bundler.Lookat(np.array([0, 0, 0]), np.array([0, 12, 20]), np.array([0, 0, 1]), DEFAULT_CALIB),
    1: PinholeCameraCal3Bundler.Lookat(np.array([1, 0, 0]), np.array([0, 12, 20]), np.array([0, 0, 1]), DEFAULT_CALIB),
}


def generate_measurements_for_tracks(
    points_3d: List[np.ndarray], cameras: Dict[int, PinholeCameraCal3Bundler], noise_sigma: float = 0.0
) -> List[SfmTrack2d]:
    """Generate measurements using projection.

    Args:
        points_3d: the points to project for each camera.
        cameras: cameras observing the points.
        noise_sigma (optional): standard deviation of the Gaussian measurement noise to add to each measurement.
                                Defaults to 0.0.

    Returns:
        Tracks with the measurement for each point.
    """

    tracks: List[SfmTrack2d] = []
    for point in points_3d:
        measurements: List[SfmMeasurement] = []
        for i, cam in cameras.items():
            measurements.append(SfmMeasurement(i, cam.project(point) + np.random.randn((2)) * noise_sigma))

        tracks.append(SfmTrack2d(measurements))

    return tracks


class TestSimpleTriangulation(unittest.TestCase):
    """Unit tests for simple triangulation."""

    def test_nonoise_with_camera_pair_good(self):
        """Tests the good pair of cameras with no measurement noise."""
        tracks_2d = generate_measurements_for_tracks(LANDMARK_POINTS_3D, CAMERA_PAIR_GOOD)

        point3d_initializer = Point3dInitializer(CAMERA_PAIR_GOOD, TriangulationParam.NO_RANSAC, reproj_error_thresh=5)

        for idx, track_2d in enumerate(tracks_2d):
            track_3d, _, _ = point3d_initializer.triangulate(track_2d)

            self.assertIsNotNone(track_3d)
            np.testing.assert_allclose(track_3d.point3(), LANDMARK_POINTS_3D[idx], atol=1e-5)

    def test_nonoise_with_camera_pair_small_baseline(self):
        """Tests the small-baseline pair of cameras with no measurement noise."""
        tracks_2d = generate_measurements_for_tracks(LANDMARK_POINTS_3D, CAMERA_PAIR_SMALL_BASELINE)

        point3d_initializer = Point3dInitializer(
            CAMERA_PAIR_SMALL_BASELINE, TriangulationParam.NO_RANSAC, reproj_error_thresh=5
        )

        for idx, track_2d in enumerate(tracks_2d):
            track_3d, _, _ = point3d_initializer.triangulate(track_2d)

            self.assertIsNotNone(track_3d)
            np.testing.assert_allclose(track_3d.point3(), LANDMARK_POINTS_3D[idx], atol=1e-5)

    def test_2pxnoise_with_camera_pair_good(self):
        """Tests the good pair of cameras with 2px measurement noise."""
        tracks_2d = generate_measurements_for_tracks(LANDMARK_POINTS_3D, CAMERA_PAIR_GOOD, noise_sigma=2)

        point3d_initializer = Point3dInitializer(CAMERA_PAIR_GOOD, TriangulationParam.NO_RANSAC, reproj_error_thresh=5)

        for idx, track_2d in enumerate(tracks_2d):
            track_3d, _, _ = point3d_initializer.triangulate(track_2d)

            self.assertIsNotNone(track_3d)
            np.testing.assert_allclose(track_3d.point3(), LANDMARK_POINTS_3D[idx], atol=1e-5)

    def test_2pxnoise_with_camera_pair_small_baseline(self):
        """Tests the small-baseline pair of cameras with 2px measurement noise."""
        tracks_2d = generate_measurements_for_tracks(LANDMARK_POINTS_3D, CAMERA_PAIR_SMALL_BASELINE, noise_sigma=2)

        point3d_initializer = Point3dInitializer(
            CAMERA_PAIR_SMALL_BASELINE, TriangulationParam.NO_RANSAC, reproj_error_thresh=5
        )

        for idx, track_2d in enumerate(tracks_2d):
            track_3d, _, _ = point3d_initializer.triangulate(track_2d)

            self.assertIsNone(track_3d)


if __name__ == "__main__":
    unittest.main()
