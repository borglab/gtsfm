"""Tests for the export-time triangulation-angle track filter."""

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, SfmTrack

from gtsfm.cluster_merging import filter_tracks_by_triangulation_angle
from gtsfm.common.gtsfm_data import GtsfmData


def _make_scene() -> GtsfmData:
    """3 cameras on the x-axis; one low-parallax track and one wide-baseline track."""
    cal = Cal3Bundler(500.0, 0.0, 0.0, 0.0, 0.0)
    scene = GtsfmData(number_images=3)
    scene.add_camera(0, PinholeCameraCal3Bundler(Pose3(Rot3(), Point3(0.0, 0.0, 0.0)), cal))
    scene.add_camera(1, PinholeCameraCal3Bundler(Pose3(Rot3(), Point3(0.01, 0.0, 0.0)), cal))
    scene.add_camera(2, PinholeCameraCal3Bundler(Pose3(Rot3(), Point3(5.0, 0.0, 0.0)), cal))

    # ~0.006 deg apex angle from cams {0,1}: depth-unconstrained.
    low_parallax = SfmTrack(np.array([0.0, 0.0, 100.0]))
    low_parallax.addMeasurement(0, np.array([0.0, 0.0]))
    low_parallax.addMeasurement(1, np.array([-0.05, 0.0]))

    # ~28 deg apex angle from cams {0,2}: well-constrained.
    wide = SfmTrack(np.array([2.5, 0.0, 10.0]))
    wide.addMeasurement(0, np.array([125.0, 0.0]))
    wide.addMeasurement(2, np.array([-125.0, 0.0]))

    assert scene.add_track(low_parallax)
    assert scene.add_track(wide)
    return scene


def test_low_parallax_track_dropped_wide_kept() -> None:
    scene = _make_scene()
    filtered = filter_tracks_by_triangulation_angle(scene, min_angle_deg=1.5)
    assert filtered.number_tracks() == 1
    kept = filtered.get_track(0)
    np.testing.assert_allclose(np.asarray(kept.point3()), [2.5, 0.0, 10.0])
    # Cameras are untouched: this is a track filter, not a camera filter.
    assert filtered.get_valid_camera_indices() == scene.get_valid_camera_indices()


def test_zero_threshold_is_identity() -> None:
    scene = _make_scene()
    assert filter_tracks_by_triangulation_angle(scene, min_angle_deg=0.0) is scene


def test_all_tracks_survive_high_parallax() -> None:
    scene = _make_scene()
    # Both tracks clear a 0.001 deg bar.
    filtered = filter_tracks_by_triangulation_angle(scene, min_angle_deg=0.001)
    assert filtered.number_tracks() == 2
