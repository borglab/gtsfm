"""Regression coverage for finite axis alignment of sparse or degenerate scenes."""

import pickle

import gtsam
import numpy as np
import pytest

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils.ellipsoid import get_ortho_axis_alignment_transform


@pytest.mark.parametrize(
    "points",
    [
        np.empty((0, 3)),
        np.array([[1.0, 2.0, 5.0]]),
        np.array([[1.0, 0.0, 5.0], [-1.0, 0.0, 5.0]]),
        np.array([[1.0, 1.0, 5.0], [-1.0, 1.0, 5.0], [1.0, -1.0, 5.0], [-1.0, -1.0, 5.0]]),
    ],
    ids=["empty", "one-point-skydio-crash", "two-points", "equal-radius-cloud"],
)
def test_sparse_scene_alignment_remains_finite_and_serializable(points: np.ndarray) -> None:
    """Axis alignment must not turn finite reconstruction data into unpicklable NaNs."""
    scene = GtsfmData(3)
    calibration = gtsam.Cal3Bundler(100.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(3):
        pose = gtsam.Pose3(gtsam.Rot3(), np.array([float(i), 0.0, 0.0]))
        scene.add_camera(i, gtsam.PinholeCameraCal3Bundler(pose, calibration))
    for point in points:
        track = gtsam.SfmTrack(point)
        for i, camera in scene.cameras().items():
            track.addMeasurement(i, camera.project(point))
        scene.add_track(track)

    transform = get_ortho_axis_alignment_transform(scene)
    assert np.isfinite(transform.matrix()).all()
    aligned = scene.transform_with_sim3(gtsam.Similarity3(transform.rotation(), transform.translation(), 1.0))
    recovered = pickle.loads(pickle.dumps(aligned))
    assert recovered.number_tracks() == len(points)
    assert all(np.isfinite(camera.pose().matrix()).all() for camera in recovered.cameras().values())
    assert all(np.isfinite(track.point3()).all() for track in recovered.tracks())
