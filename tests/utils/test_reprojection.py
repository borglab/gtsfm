"""
Unit tests for reprojection utilities.
"""

from typing import Dict, cast

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3, SfmTrack  # type: ignore

import gtsfm.utils.reprojection as reproj_utils
from gtsfm.common.sfm_track import SfmMeasurement
from gtsfm.common.types import CAMERA_TYPE


# Helper function and constants to match GTSAM C++ style
def vector(arr) -> np.ndarray:
    """Helper to create properly typed arrays for GTSAM functions."""
    return np.array(arr, dtype=np.float64)


def camera_dict(**cameras) -> Dict[int, CAMERA_TYPE]:
    """Helper to create properly typed camera dictionaries."""
    return cast(Dict[int, CAMERA_TYPE], cameras)


Z_3x1: np.ndarray = vector([0, 0, 0])


def test_compute_track_reprojection_errors():
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
    wTi0 = Pose3(Rot3.RzRyRx(0, 0, 0), Z_3x1)
    wTi1 = Pose3(Rot3.RzRyRx(0, 0, 0), vector([2, -2, 0]))

    f = 10
    k1 = 0
    k2 = 0
    u0 = 3
    v0 = 4

    K0 = Cal3Bundler(f, k1, k2, u0, v0)
    K1 = Cal3Bundler(f, k1, k2, u0, v0)

    track_camera_dict = {0: PinholeCameraCal3Bundler(wTi0, K0), 1: PinholeCameraCal3Bundler(wTi1, K1)}

    track_3d = SfmTrack(vector([1, 2, 1]))

    # in camera 0
    track_3d.addMeasurement(idx=0, m=vector([13, 24]))
    # in camera 1
    track_3d.addMeasurement(idx=1, m=vector([-8, 43]))  # should be (-7,44), 1 px error in each dim

    errors, avg_track_reproj_error = reproj_utils.compute_track_reprojection_errors(
        track_camera_dict, track_3d  # type: ignore
    )

    expected_errors = np.array([0, np.sqrt(2)])
    np.testing.assert_allclose(errors, expected_errors)
    assert avg_track_reproj_error == np.sqrt(2) / 2


def test_compute_point_reprojection_errors():
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
    wTi0 = Pose3(Rot3.RzRyRx(0, 0, 0), Z_3x1)
    wTi1 = Pose3(Rot3.RzRyRx(0, 0, 0), vector([2, -2, 0]))

    f = 10
    k1 = 0
    k2 = 0
    u0 = 3
    v0 = 4

    K0 = Cal3Bundler(f, k1, k2, u0, v0)
    K1 = Cal3Bundler(f, k1, k2, u0, v0)

    track_camera_dict = {0: PinholeCameraCal3Bundler(wTi0, K0), 1: PinholeCameraCal3Bundler(wTi1, K1)}
    point3d = np.array([1, 2, 1])
    measurements = [SfmMeasurement(i=1, uv=np.array([-8, 43])), SfmMeasurement(i=0, uv=np.array([13, 24]))]

    errors, avg_track_reproj_error = reproj_utils.compute_point_reprojection_errors(
        track_camera_dict, point3d, measurements  # type: ignore
    )
    expected_errors = np.array([np.sqrt(2), 0])
    np.testing.assert_allclose(errors, expected_errors)
    assert avg_track_reproj_error == np.sqrt(2) / 2


def test_compute_point_reprojection_errors_missing_camera():
    """Test handling of measurements with missing cameras (should produce NaN).

    This tests the edge case where a measurement references a camera that doesn't
    exist in track_camera_dict, which can happen when camera pose estimation fails.
    """
    wTi0 = Pose3(Rot3.RzRyRx(0, 0, 0), Z_3x1)
    K0 = Cal3Bundler(10, 0, 0, 3, 4)
    track_camera_dict = {0: PinholeCameraCal3Bundler(wTi0, K0)}  # Only camera 0

    point3d = np.array([1, 2, 1])
    measurements = [
        SfmMeasurement(i=0, uv=np.array([13, 24])),  # Valid camera
        SfmMeasurement(i=999, uv=np.array([0, 0])),  # Missing camera 999
    ]

    errors, avg_error = reproj_utils.compute_point_reprojection_errors(
        track_camera_dict, point3d, measurements  # type: ignore
    )

    # First error should be 0 (perfect projection), second should be NaN
    assert errors[0] == 0.0
    assert np.isnan(errors[1])
    # Average should only consider valid measurements
    assert avg_error == 0.0


def test_compute_point_reprojection_errors_projection_failure():
    """Test handling of failed projections (point behind camera)."""
    wTi0 = Pose3(Rot3.RzRyRx(0, 0, 0), Z_3x1)
    K0 = Cal3Bundler(10, 0, 0, 3, 4)
    track_camera_dict = {0: PinholeCameraCal3Bundler(wTi0, K0)}

    point_behind_camera = np.array([0, 0, -1])  # Negative Z = behind camera
    measurements = [SfmMeasurement(i=0, uv=np.array([13, 24]))]

    errors, avg_error = reproj_utils.compute_point_reprojection_errors(
        track_camera_dict, point_behind_camera, measurements  # type: ignore
    )

    # Should produce NaN for failed projection
    assert np.isnan(errors[0])
    # Average should be NaN when all errors are NaN
    assert np.isnan(avg_error)


def test_compute_track_reprojection_errors_mixed_validity():
    """Test track with mix of valid and invalid measurements."""
    wTi0 = Pose3(Rot3.RzRyRx(0, 0, 0), Z_3x1)
    K0 = Cal3Bundler(10, 0, 0, 3, 4)
    track_camera_dict = {0: PinholeCameraCal3Bundler(wTi0, K0)}  # Only camera 0

    triangulated_pt = vector([1, 2, 1])
    track_3d = SfmTrack(triangulated_pt)

    # Add measurement for existing camera (should work)
    track_3d.addMeasurement(idx=0, m=vector([13, 24]))
    # Add measurement for non-existent camera (should produce NaN)
    track_3d.addMeasurement(idx=999, m=vector([0, 0]))

    errors, avg_error = reproj_utils.compute_track_reprojection_errors(track_camera_dict, track_3d)  # type: ignore

    # First error should be 0 (perfect projection), second should be NaN
    assert errors[0] == 0.0
    assert np.isnan(errors[1])
    # Average should only consider valid measurements
    assert avg_error == 0.0
