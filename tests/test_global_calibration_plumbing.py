"""Tests for the global-calibration plumbing: sparse refined-intrinsics fallback in the cluster build.

The global view-graph calibration hands every cluster a (possibly sparse) dict of per-camera focals via
``ClusterContext.global_refined_intrinsics``. Cameras present in the dict are pinned verbatim; cameras
absent from it must fall back to the geometry model's predicted focal, rescaled to original resolution.

Authors: Kathirvel Gounder
"""

import unittest

import numpy as np
from gtsam import Cal3Bundler, Cal3_S2, PinholeCameraCal3Bundler, PinholeCameraCal3_S2, Pose3

from gtsfm.cluster_optimizer.cluster_vggt_with_frontend import (
    ClusterVGGTWithFrontend,
    VggtGeometryResult,
    _build_gtsfm_data_from_vggt_depth,
    _scale_camera_intrinsics,
)


class TestSparseRefinedIntrinsicsFallback(unittest.TestCase):
    def test_absent_cameras_fall_back_to_model_focal(self) -> None:
        num_local, H, W = 2, 8, 8
        model_cal = Cal3Bundler(100.0, 0.0, 0.0, 4.0, 4.0)  # in model pixel space (scaled width = 8)
        cameras = {
            0: PinholeCameraCal3Bundler(Pose3(), model_cal),
            1: PinholeCameraCal3Bundler(Pose3(), model_cal),
        }
        original_coords = np.zeros((num_local, 6), dtype=np.float32)
        original_coords[:, 4] = W
        original_coords[:, 5] = H
        vggt_result = VggtGeometryResult(
            cameras=cameras,
            dense_points=np.zeros((num_local, H, W, 3), dtype=np.float32),
            depth_confidence=np.ones((num_local, H, W), dtype=np.float32),
            original_coords=original_coords,
        )
        image_shapes = {0: (16, 16), 1: (16, 16)}  # original resolution = 2x the model resolution
        refined_cal = Cal3Bundler(555.0, 0.0, 0.0, 8.0, 8.0)

        data = _build_gtsfm_data_from_vggt_depth(
            vggt_result,
            tracks_2d=[],
            image_shapes=image_shapes,
            image_indices=(0, 1),
            num_images=2,
            min_track_length=2,
            refined_intrinsics={0: refined_cal},
        )

        self.assertAlmostEqual(data.get_camera(0).calibration().fx(), 555.0)  # pinned verbatim
        self.assertAlmostEqual(data.get_camera(1).calibration().fx(), 200.0)  # model 100 x (16/8)

    def test_calibration_flags_enter_cache_key_repr(self) -> None:
        """The cluster cache key is seeded from repr(optimizer); the calibration flags must appear."""
        from unittest.mock import MagicMock

        optimizer = ClusterVGGTWithFrontend(
            correspondence_generator=MagicMock(),
            two_view_estimator=MagicMock(),
            use_global_view_graph_calibration=True,
        )
        self.assertIn("calib=(vgc=False,global=True)", repr(optimizer))
        self.assertTrue(optimizer.uses_global_view_graph_calibration)


class TestAnisotropicIntrinsicScale(unittest.TestCase):
    def test_principal_point_follows_per_axis_scale(self) -> None:
        # Resized frame is 40 x 80. The principal point is not its center.
        cal = Cal3_S2(120.0, 80.0, 0.0, 30.0, 10.0)
        scaled = _scale_camera_intrinsics(
            PinholeCameraCal3_S2(Pose3(), cal), scale_x=2.0, scale_y=4.0, use_cal3_bundler=True
        )
        out = scaled.calibration()

        self.assertIsInstance(out, Cal3Bundler)
        self.assertAlmostEqual(out.fx(), (240.0 + 320.0) / 2.0)
        self.assertAlmostEqual(out.px(), 60.0)
        self.assertAlmostEqual(out.py(), 40.0)

    def test_depth_build_uses_height_ratio_for_cy(self) -> None:
        cal = Cal3_S2(100.0, 50.0, 0.0, 10.0, 20.0)
        original_coords = np.zeros((1, 6), dtype=np.float32)
        original_coords[0, 4] = 40.0  # scaled_w
        original_coords[0, 5] = 80.0  # scaled_h
        vggt_result = VggtGeometryResult(
            cameras={0: PinholeCameraCal3_S2(Pose3(), cal)},
            dense_points=np.zeros((1, 8, 8, 3), dtype=np.float32),
            depth_confidence=np.ones((1, 8, 8), dtype=np.float32),
            original_coords=original_coords,
        )

        data = _build_gtsfm_data_from_vggt_depth(
            vggt_result,
            tracks_2d=[],
            image_shapes={0: (80, 80)},  # scale_x = 2, scale_y = 1
            image_indices=(0,),
            num_images=1,
            use_cal3_bundler=True,
        )
        out = data.get_camera(0).calibration()

        self.assertAlmostEqual(out.fx(), 125.0)  # (100*2 + 50*1) / 2
        self.assertAlmostEqual(out.px(), 20.0)
        self.assertAlmostEqual(out.py(), 20.0)

    def test_default_keeps_separate_focals(self) -> None:
        cal = Cal3_S2(120.0, 80.0, 0.0, 30.0, 10.0)
        scaled = _scale_camera_intrinsics(PinholeCameraCal3_S2(Pose3(), cal), scale_x=2.0, scale_y=4.0)
        out = scaled.calibration()

        self.assertIsInstance(out, Cal3_S2)
        self.assertAlmostEqual(out.fx(), 240.0)
        self.assertAlmostEqual(out.fy(), 320.0)
        self.assertAlmostEqual(out.px(), 60.0)
        self.assertAlmostEqual(out.py(), 40.0)


if __name__ == "__main__":
    unittest.main()
