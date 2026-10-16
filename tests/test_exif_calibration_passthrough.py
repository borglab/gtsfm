"""Tests for the measured-intrinsics passthrough and per-camera calibration arbitration.

Cameras whose intrinsics were measured (EXIF / dataset calibration) are passed through the
SceneOptimizer verbatim and pinned in the cluster builds; cameras whose loader could only guess a
focal are excluded and fall back to the geometry model's predicted focal, rescaled to original
resolution.

Authors: Kathirvel Gounder
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3

from gtsfm.cluster_optimizer.cluster_optimizer_cacher import ClusterOptimizerCacher
from gtsfm.cluster_optimizer.cluster_vggt_with_frontend import VggtGeometryResult, _build_gtsfm_data_from_vggt_depth
from gtsfm.products.one_view_data import OneViewData
from gtsfm.scene_optimizer import exif_intrinsics_passthrough


def _one_view_data(fx: float, from_exif: bool) -> OneViewData:
    return OneViewData(
        image_fname="img.jpg",
        intrinsics=Cal3Bundler(fx, 0.0, 0.0, 100.0, 75.0),
        absolute_pose_prior=None,
        camera_gt=None,
        pose_gt=None,
        intrinsics_from_exif=from_exif,
    )


class TestExifIntrinsicsPassthrough(unittest.TestCase):
    def test_only_measured_cameras_pass_through(self) -> None:
        one_view_data_dict = {
            0: _one_view_data(800.0, from_exif=True),
            1: _one_view_data(900.0, from_exif=False),
            2: _one_view_data(1000.0, from_exif=True),
        }
        passthrough = exif_intrinsics_passthrough(one_view_data_dict)
        self.assertEqual(set(passthrough.keys()), {0, 2})
        self.assertEqual(passthrough[0].fx(), 800.0)
        self.assertEqual(passthrough[2].fx(), 1000.0)

    def test_default_provenance_is_measured(self) -> None:
        """Loaders that never guess need no override: the default marks intrinsics as measured."""
        one_view_data = OneViewData(
            image_fname="img.jpg",
            intrinsics=Cal3Bundler(800.0, 0.0, 0.0, 100.0, 75.0),
            absolute_pose_prior=None,
            camera_gt=None,
            pose_gt=None,
        )
        self.assertTrue(one_view_data.intrinsics_from_exif)

    def test_provenance_changes_cluster_cache_key(self) -> None:
        """Pinning toggles cluster output, so provenance must participate in the cache key."""
        cacher = ClusterOptimizerCacher(optimizer=MagicMock())
        hash_measured = cacher._hash_one_view_data(_one_view_data(800.0, from_exif=True))
        hash_guessed = cacher._hash_one_view_data(_one_view_data(800.0, from_exif=False))
        self.assertNotEqual(hash_measured, hash_guessed)


class TestBuildCalibrationArbitration(unittest.TestCase):
    def test_sparse_refined_intrinsics_fall_back_to_model_focal(self) -> None:
        """Cameras absent from refined_intrinsics keep the model focal, rescaled to original res."""
        num_local, H, W = 2, 8, 8
        model_cal = Cal3Bundler(100.0, 0.0, 0.0, 4.0, 4.0)  # in VGGT pixel space (scaled width = 8)
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
        image_shapes = {0: (16, 16), 1: (16, 16)}  # original resolution = 2x the VGGT resolution
        measured_cal = Cal3Bundler(555.0, 0.0, 0.0, 8.0, 8.0)

        data = _build_gtsfm_data_from_vggt_depth(
            vggt_result,
            tracks_2d=[],
            image_shapes=image_shapes,
            image_indices=(0, 1),
            num_images=2,
            min_track_length=2,
            refined_intrinsics={0: measured_cal},
        )

        self.assertAlmostEqual(data.get_camera(0).calibration().fx(), 555.0)  # pinned verbatim
        self.assertAlmostEqual(data.get_camera(1).calibration().fx(), 200.0)  # model 100 x (16/8)


if __name__ == "__main__":
    unittest.main()
