"""Tests for the run_bundle_adjustment=False path of _run_cluster_ba."""

import sys
import types

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, SfmTrack

# cluster_vggt imports the VGGT model stack at module level; stub it so this CPU-only unit
# test (which only exercises the BA wrapper) can import without the vggt package installed.
if "vggt" not in sys.modules:
    for name in ("vggt", "vggt.models", "vggt.models.vggt", "vggt.utils",
                 "vggt.utils.geometry", "vggt.utils.helper", "vggt.utils.load_fn",
                 "vggt.utils.pose_enc"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vggt.models.vggt"].VGGT = object
    sys.modules["vggt.utils.geometry"].unproject_depth_map_to_point_map = lambda *a, **k: None
    sys.modules["vggt.utils.helper"].randomly_limit_trues = lambda *a, **k: None
    sys.modules["vggt.utils.load_fn"].load_and_preprocess_images_square = lambda *a, **k: None
    sys.modules["vggt.utils.pose_enc"].pose_encoding_to_extri_intri = lambda *a, **k: None

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptions
from gtsfm.cluster_optimizer.cluster_vggt import _run_cluster_ba
from gtsfm.common.gtsfm_data import GtsfmData


def _make_scene() -> GtsfmData:
    cal = Cal3Bundler(500.0, 0.0, 0.0, 0.0, 0.0)
    scene = GtsfmData(number_images=2)
    scene.add_camera(0, PinholeCameraCal3Bundler(Pose3(Rot3(), Point3(0.0, 0.0, 0.0)), cal))
    scene.add_camera(1, PinholeCameraCal3Bundler(Pose3(Rot3(), Point3(5.0, 0.0, 0.0)), cal))
    track = SfmTrack(np.array([2.5, 0.0, 10.0]))
    track.addMeasurement(0, np.array([125.0, 0.0]))
    track.addMeasurement(1, np.array([-125.0, 0.0]))
    assert scene.add_track(track)
    return scene


def test_ba_off_keeps_poses_and_tracks_verbatim() -> None:
    scene = _make_scene()
    post, pre = _run_cluster_ba(
        scene,
        ba_options=BundleAdjustmentOptions(),
        pre_ba_max_reproj_error=14.0,
        min_track_length=2,
        run_bundle_adjustment=False,
    )
    # Poses untouched (no optimizer ran), tracks retained (0px reproj passes the 14px pre-filter).
    for i in (0, 1):
        np.testing.assert_allclose(
            np.asarray(post.get_camera(i).pose().translation()),
            np.asarray(scene.get_camera(i).pose().translation()),
        )
    assert post.number_tracks() == 1
    assert pre.number_tracks() == 1


def test_ba_off_still_applies_pre_ba_filter() -> None:
    scene = _make_scene()
    bad = SfmTrack(np.array([2.5, 0.0, 10.0]))
    bad.addMeasurement(0, np.array([300.0, 200.0]))  # ~180px reproj error: junk
    bad.addMeasurement(1, np.array([-300.0, -200.0]))
    assert scene.add_track(bad)
    post, _ = _run_cluster_ba(
        scene,
        ba_options=BundleAdjustmentOptions(),
        pre_ba_max_reproj_error=14.0,
        min_track_length=2,
        run_bundle_adjustment=False,
    )
    assert post.number_tracks() == 1  # junk track filtered, good one kept
