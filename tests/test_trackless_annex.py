"""Unit tests for the prior-backed trackless-camera ANNEX (export_trackless_annex).

The annex carries filter-dropped cameras up the merge tree OUTSIDE the scenes: re-seated by the same
Sim3 that seats their subtree, never entering a Sim3 solve, duplicate resolution, or BA. These tests
pin the two contracts that make it safe:
  1. carry math — a child's annex comes out transformed by exactly the child->merged Sim3;
  2. inertness — flag on/off produces an IDENTICAL merged scene (the annex is pure extra output).

Authors: Kathirvel Gounder
"""

import unittest

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, SfmTrack, Similarity3

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptions
from gtsfm.cluster_merging import (
    MergedNodeResult,
    MergingOptions,
    _carry_annex_cameras,
    combine_results,
)
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.metrics import GtsfmMetricsGroup

NUM_IMAGES = 32
CALIB = Cal3Bundler(500.0, 0.0, 0.0, 320.0, 240.0)


def _cam(x: float, y: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> PinholeCameraCal3Bundler:
    pose = Pose3(Rot3.Yaw(yaw), np.array([x, y, z]))
    return PinholeCameraCal3Bundler(pose, CALIB)


def _empty_metrics() -> GtsfmMetricsGroup:
    return GtsfmMetricsGroup("test", [])


def _result(scene, annex=None) -> MergedNodeResult:
    return MergedNodeResult(
        scene=scene,
        pre_ba_scene=None,
        metrics=_empty_metrics(),
        pre_ba_metrics=_empty_metrics(),
        annex_cameras=annex,
    )


def _pose_close(a: Pose3, b: Pose3, tol: float = 1e-5) -> bool:
    return np.allclose(a.matrix(), b.matrix(), atol=tol)


class TestCarryAnnexCameras(unittest.TestCase):
    """Direct contracts of _carry_annex_cameras (no BA involved)."""

    def setUp(self) -> None:
        # Child frame b: four non-degenerate cameras (carry guards need >= 4 non-collinear
        # anchors) + one annex camera.
        self.child = GtsfmData(number_images=NUM_IMAGES)
        for idx, (x, y, z, yaw) in zip(
            (10, 11, 12, 14), [(0, 0, 0, 0.0), (4, 0, 0, 0.3), (0, 3, 1, -0.2), (2, 1, 4, 0.1)]
        ):
            self.child.add_camera(idx, _cam(x, y, z, yaw))
        self.annex_cam = _cam(7.0, 1.0, 2.0, 0.5)
        # A known Sim3 aSb seating child frame b into merged frame a.
        self.aSb = Similarity3(Rot3.Yaw(0.7), np.array([1.0, 2.0, 3.0]), 2.0)

    def _merged_from_child(self, extra_ids=()) -> GtsfmData:
        merged = GtsfmData(number_images=NUM_IMAGES)
        for i in self.child.get_valid_camera_indices():
            cam_b = self.child.get_camera(i)
            merged.add_camera(i, PinholeCameraCal3Bundler(self.aSb.transformFrom(cam_b.pose()), CALIB))
        for i in extra_ids:
            merged.add_camera(i, _cam(20.0 + i))
        return merged

    def test_carry_reseats_child_annex_by_solved_sim3(self) -> None:
        merged = self._merged_from_child()
        child_result = _result(self.child, annex={13: self.annex_cam})
        annex = _carry_annex_cameras(merged, (child_result,), {})
        self.assertIsNotNone(annex)
        self.assertIn(13, annex)
        expected = self.aSb.transformFrom(self.annex_cam.pose())
        self.assertTrue(_pose_close(annex[13].pose(), expected))
        # Calibration rides along unscaled.
        self.assertAlmostEqual(annex[13].calibration().fx(), CALIB.fx(), places=6)

    def test_carry_purges_reregistered_camera(self) -> None:
        # Camera 13 re-registered with tracks in the merged scene: the scene copy wins.
        merged = self._merged_from_child(extra_ids=(13,))
        child_result = _result(self.child, annex={13: self.annex_cam})
        annex = _carry_annex_cameras(merged, (child_result,), {})
        self.assertIsNone(annex)

    def test_carry_skips_child_with_thin_overlap(self) -> None:
        # Only 3 shared cameras: below the 4-anchor floor — annex dropped.
        merged = self._merged_from_child()
        thin_child = GtsfmData(number_images=NUM_IMAGES)
        for i in (10, 11, 12):
            thin_child.add_camera(i, self.child.get_camera(i))
        child_result = _result(thin_child, annex={13: self.annex_cam})
        annex = _carry_annex_cameras(merged, (child_result,), {})
        self.assertIsNone(annex)

    def test_carry_skips_collinear_anchors(self) -> None:
        # 4 shared cameras on a line: the Sim3 fit's scale/rotation are unconstrained around the
        # axis (the RF annex explosion) — annex dropped.
        line_child = GtsfmData(number_images=NUM_IMAGES)
        for k, i in enumerate((10, 11, 12, 14)):
            line_child.add_camera(i, _cam(float(k), 0.0, 0.0))
        merged = GtsfmData(number_images=NUM_IMAGES)
        for i in line_child.get_valid_camera_indices():
            cam_b = line_child.get_camera(i)
            merged.add_camera(i, PinholeCameraCal3Bundler(self.aSb.transformFrom(cam_b.pose()), CALIB))
        child_result = _result(line_child, annex={13: self.annex_cam})
        annex = _carry_annex_cameras(merged, (child_result,), {})
        self.assertIsNone(annex)

    def test_fresh_drops_override_child_annex(self) -> None:
        merged = self._merged_from_child()
        stale = _cam(50.0)
        fresh = _cam(60.0)
        child_result = _result(self.child, annex={13: stale})
        annex = _carry_annex_cameras(merged, (child_result,), {13: fresh})
        self.assertIsNotNone(annex)
        self.assertTrue(_pose_close(annex[13].pose(), fresh.pose()))


class TestCombineResultsAnnex(unittest.TestCase):
    """End-to-end combine_results: the post-BA filter drop is captured, and the scene is untouched."""

    def _build_scenes(self):
        # 12 points seen by everything; consistent geometry (uv = exact projections).
        points = [Point3(float(px), float(py), 6.0 + 0.2 * px) for px in range(-3, 3) for py in (-1, 1)]
        parent_cams = {0: _cam(0.0), 1: _cam(1.0), 2: _cam(2.0)}
        child_cams = {1: _cam(1.0), 2: _cam(2.0), 11: _cam(3.0), 13: _cam(4.0)}

        def scene_with(cams: dict, track_specs) -> GtsfmData:
            data = GtsfmData(number_images=NUM_IMAGES)
            for i, c in cams.items():
                data.add_camera(i, c)
            for pt, cam_ids in track_specs:
                track = SfmTrack(pt)
                for i in cam_ids:
                    track.addMeasurement(i, cams[i].project(pt))
                assert data.add_track(track)
            return data

        parent = scene_with(parent_cams, [(p, [0, 1, 2]) for p in points])
        # Child: same 12 points from cams {1,2,11} (fuses with parent tracks via shared measurement
        # keys), plus ONE 2-view victim track that is camera 13's only support.
        victim_point = Point3(3.5, 0.0, 7.0)
        child = scene_with(
            child_cams,
            [(p, [1, 2, 11]) for p in points] + [(victim_point, [11, 13])],
        )
        return parent, child

    def _options(self, export_annex: bool) -> MergingOptions:
        return MergingOptions(
            run_bundle_adjustment=True,
            merge_duplicate_tracks=True,
            drop_outlier_after_camera_merging=False,
            drop_camera_with_no_track=False,
            keep_all_cameras=False,
            export_trackless_annex=export_annex,
            use_nonlinear_sim3_alignment=True,
            pre_ba_max_reproj_error=0.0,  # keep the 2-view victim alive until the post-BA filter
            post_ba_max_reproj_error=3.0,
            min_track_length=3,
            allow_post_ba_reproj_filtering=True,
            plot_reprojection_histograms=False,
            ba_options=BundleAdjustmentOptions(
                use_pose_prior_all_cameras=True,
                min_tracks_per_camera=1,
            ),
        )

    def test_capture_and_inertness(self) -> None:
        parent, child = self._build_scenes()
        result_off = combine_results(parent, (_result(child),), options=self._options(False))
        parent2, child2 = self._build_scenes()
        result_on = combine_results(parent2, (_result(child2),), options=self._options(True))

        # Flag off: camera 13 (2-view track only) dies silently at the post-BA filter.
        self.assertIsNone(result_off.annex_cameras)
        off_ids = set(result_off.scene.get_valid_camera_indices())
        self.assertNotIn(13, off_ids)
        self.assertEqual(off_ids, {0, 1, 2, 11})

        # Flag on: IDENTICAL scene (inertness) ...
        on_ids = set(result_on.scene.get_valid_camera_indices())
        self.assertEqual(on_ids, off_ids)
        for i in off_ids:
            self.assertTrue(
                _pose_close(result_off.scene.get_camera(i).pose(), result_on.scene.get_camera(i).pose())
            )
        self.assertEqual(result_off.scene.number_tracks(), result_on.scene.number_tracks())

        # ... plus camera 13 captured in the annex near its input pose (pose priors anchor the BA).
        self.assertIsNotNone(result_on.annex_cameras)
        self.assertIn(13, result_on.annex_cameras)
        self.assertTrue(
            _pose_close(result_on.annex_cameras[13].pose(), _cam(4.0).pose(), tol=1e-2)
        )


if __name__ == "__main__":
    unittest.main()
