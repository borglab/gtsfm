"""Tests for the global-calibration plumbing: sparse refined-intrinsics fallback in the cluster build.

The global view-graph calibration hands every cluster a (possibly sparse) dict of per-camera focals via
``ClusterContext.global_refined_intrinsics``. Cameras present in the dict are pinned verbatim; cameras
absent from it must fall back to the geometry model's predicted focal, rescaled to original resolution.

Authors: Kathirvel Gounder
"""

import unittest

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3

from gtsfm.cluster_optimizer.cluster_vggt_with_frontend import (
    ClusterVGGTWithFrontend,
    VggtGeometryResult,
    _build_gtsfm_data_from_vggt_depth,
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


if __name__ == "__main__":
    unittest.main()
