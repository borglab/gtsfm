"""Regression for a BA solution that hides observations behind the cameras."""

from pathlib import Path

import numpy as np
import pytest
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, SfmTrack

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "skydio_ba_cheirality" / "scene.npz"


def _load_scene(track_shift: int) -> GtsfmData:
    """Load the frozen scene, optionally relabeling tracks without changing observations.

    Args:
        track_shift: Cyclic shift of the track variable indices.

    Returns:
        Scene containing the same geometry and measurements under the selected labeling.
    """
    with np.load(FIXTURE, allow_pickle=False) as data:
        cameras = {
            int(i): PinholeCameraCal3Bundler(Pose3(pose), Cal3Bundler(*calibration))
            for i, pose, calibration in zip(data["camera_ids"], data["poses"], data["calibrations"])
        }
        tracks = []
        for j, point in enumerate(data["points"]):
            track = SfmTrack(point)
            for k in range(data["offsets"][j], data["offsets"][j + 1]):
                track.addMeasurement(int(data["measurement_camera_ids"][k]), data["measurements"][k])
            tracks.append(track)
        order = np.roll(np.arange(len(tracks)), track_shift)
        return GtsfmData(32, cameras, [tracks[j] for j in order])


@pytest.mark.parametrize("track_shift", [0, 1])
def test_ba_preserves_visible_scene(track_shift: int) -> None:
    """A well-initialized scene must retain its cameras rather than minimize cost by losing visibility."""
    scene = _load_scene(track_shift)
    optimizer = BundleAdjustmentOptimizer(
        robust_ba_mode="GMC", use_calibration_prior=False, measurement_noise_sigma=2.0
    )

    optimized, _ = optimizer.run_simple_ba(scene)
    filtered = optimized.filter_landmark_measurements(3.0, 2, retain_cameras_without_tracks=False)

    assert len(filtered.cameras()) == 32
    assert filtered.number_tracks() > 3000
