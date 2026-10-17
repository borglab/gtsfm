"""Tests for the triangulated-structure cluster build (``_build_gtsfm_data_via_triangulation``).

The build keeps the geometry model's POSES but replaces its per-pixel depth lift with SIFT tracks
triangulated against those poses, so the structure is multi-view consistent by construction. These
tests pin (a) the per-camera intrinsics arbitration (refined dict pinned verbatim, absent cameras fall
back to the rescaled model focal) and (b) an actual triangulation: two cameras with exact projections
of a known 3D point must recover it.

Authors: Kathirvel Gounder
"""

import unittest

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3

from gtsfm.cluster_optimizer.cluster_vggt_with_frontend import (
    VggtGeometryResult,
    _build_gtsfm_data_via_triangulation,
)
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d


def _vggt_result(cameras, scaled_w: float) -> VggtGeometryResult:
    num_local, H, W = len(cameras), 8, 8
    original_coords = np.zeros((num_local, 6), dtype=np.float32)
    original_coords[:, 4] = scaled_w
    original_coords[:, 5] = scaled_w
    return VggtGeometryResult(
        cameras=cameras,
        dense_points=np.zeros((num_local, H, W, 3), dtype=np.float32),
        depth_confidence=np.ones((num_local, H, W), dtype=np.float32),
        original_coords=original_coords,
    )


class TestTriangulatedStructureBuild(unittest.TestCase):
    def test_intrinsics_arbitration_matches_depth_build(self) -> None:
        """Refined intrinsics pinned verbatim; absent cameras rescale the model focal."""
        model_cal = Cal3Bundler(100.0, 0.0, 0.0, 4.0, 4.0)  # model pixel space, scaled width 8
        cameras = {
            0: PinholeCameraCal3Bundler(Pose3(), model_cal),
            1: PinholeCameraCal3Bundler(Pose3(), model_cal),
        }
        data = _build_gtsfm_data_via_triangulation(
            _vggt_result(cameras, scaled_w=8.0),
            tracks_2d=[],
            image_shapes={0: (16, 16), 1: (16, 16)},  # original res = 2x model res
            image_indices=(0, 1),
            num_images=2,
            min_track_length=2,
            refined_intrinsics={0: Cal3Bundler(555.0, 0.0, 0.0, 8.0, 8.0)},
        )
        self.assertAlmostEqual(data.get_camera(0).calibration().fx(), 555.0)
        self.assertAlmostEqual(data.get_camera(1).calibration().fx(), 200.0)
        self.assertEqual(data.number_tracks(), 0)

    def test_two_camera_track_triangulates_to_known_point(self) -> None:
        """Exact projections of a known 3D point must triangulate back to it."""
        cal = Cal3Bundler(100.0, 0.0, 0.0, 0.0, 0.0)
        cam0 = PinholeCameraCal3Bundler(Pose3(), cal)
        cam1 = PinholeCameraCal3Bundler(Pose3(Rot3(), Point3(1.0, 0.0, 0.0)), cal)
        point = Point3(0.3, 0.2, 5.0)
        track = SfmTrack2d(
            measurements=[
                SfmMeasurement(0, np.asarray(cam0.project(point))),
                SfmMeasurement(1, np.asarray(cam1.project(point))),
            ]
        )
        data = _build_gtsfm_data_via_triangulation(
            _vggt_result({0: cam0, 1: cam1}, scaled_w=16.0),
            tracks_2d=[track],
            image_shapes={0: (16, 16), 1: (16, 16)},  # original res == model res: intrinsics unchanged
            image_indices=(0, 1),
            num_images=2,
            min_track_length=2,
        )
        self.assertEqual(data.number_tracks(), 1)
        recovered = data.get_track(0).point3()
        np.testing.assert_allclose(np.asarray(recovered), [0.3, 0.2, 5.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()


class TestDepthBuildMeasurementSemantics(unittest.TestCase):
    def test_zero_confidence_measurements_still_constrain_ba(self) -> None:
        """A keypoint in a zero-confidence depth region must still enter the track as a BA constraint
        (June-27 golden-run semantics); only confident depths anchor the 3D point."""
        from gtsfm.cluster_optimizer.cluster_vggt_with_frontend import _build_gtsfm_data_from_vggt_depth

        num_local, H, W = 3, 8, 8
        cal = Cal3Bundler(100.0, 0.0, 0.0, 4.0, 4.0)
        cameras = {i: PinholeCameraCal3Bundler(Pose3(), cal) for i in range(3)}
        original_coords = np.zeros((num_local, 6), dtype=np.float32)
        original_coords[:, 4] = W
        original_coords[:, 5] = H
        dense = np.ones((num_local, H, W, 3), dtype=np.float32) * 5.0
        conf = np.ones((num_local, H, W), dtype=np.float32)
        conf[2] = 0.0  # camera 2's entire depth map is zero-confidence

        from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d

        track = SfmTrack2d(measurements=[SfmMeasurement(i, np.array([4.0, 4.0], dtype=np.float32)) for i in range(3)])
        data = _build_gtsfm_data_from_vggt_depth(
            VggtGeometryResult(
                cameras=cameras, dense_points=dense, depth_confidence=conf, original_coords=original_coords
            ),
            tracks_2d=[track],
            image_shapes={i: (8, 8) for i in range(3)},
            image_indices=(0, 1, 2),
            num_images=3,
            min_track_length=2,
        )
        self.assertEqual(data.number_tracks(), 1)
        # All three measurements constrain the track — including camera 2's zero-confidence keypoint.
        self.assertEqual(data.get_track(0).numberMeasurements(), 3)
