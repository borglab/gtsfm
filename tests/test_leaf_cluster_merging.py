"""Tests for merging leaf reconstructions without a parent scene."""

import unittest

import gtsam
import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point2, Point3, Pose3, Rot3, SfmTrack

from gtsfm.cluster_merging import MergedNodeResult, MergingOptions, combine_results
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.metrics import GtsfmMetricsGroup


def _camera(translation: tuple[float, float, float]) -> PinholeCameraCal3Bundler:
    calibration = Cal3Bundler(100.0, 0.0, 0.0, 0.0, 0.0)
    return PinholeCameraCal3Bundler(Pose3(Rot3(), Point3(*translation)), calibration)


def _track(point: tuple[float, float, float], measurements: list[tuple[int, tuple[float, float]]]) -> SfmTrack:
    track = SfmTrack(Point3(*point))
    for camera_idx, uv in measurements:
        track.addMeasurement(camera_idx, Point2(*uv))
    return track


def _scene(cameras: dict[int, tuple[float, float, float]], tracks: list[SfmTrack]) -> GtsfmData:
    scene = GtsfmData(number_images=max(cameras) + 1)
    for index, translation in cameras.items():
        scene.add_camera(index, _camera(translation))
    for track in tracks:
        scene.add_track(track)
    return scene


def _child_result(scene: GtsfmData) -> MergedNodeResult:
    metrics = GtsfmMetricsGroup("unused", [])
    return MergedNodeResult(scene=scene, pre_ba_scene=None, metrics=metrics, pre_ba_metrics=metrics)


class TestLeafClusterMerging(unittest.TestCase):
    def test_children_align_by_shared_tracks_without_parent(self) -> None:
        shared = [((0.0, 0.0, 5.0), (10.0, 10.0)), ((1.0, 0.0, 5.0), (20.0, 10.0)), ((0.0, 1.0, 5.0), (10.0, 20.0))]
        shift = np.array([2.0, 0.0, 0.0])
        reference = _scene(
            {0: (0.0, 0.0, 0.0)},
            [_track(point, [(0, uv)]) for point, uv in shared],
        )
        other_tracks = [_track(tuple(np.array(point) + shift), [(0, uv)]) for point, uv in shared]
        other_tracks.append(_track((5.0, 1.0, 4.0), [(1, (30.0, 40.0))]))
        other = _scene({0: (0.0, 0.0, 0.0), 1: (8.0, 0.0, 0.0)}, other_tracks)

        options = MergingOptions(
            reconstruct_internal_nodes=False,
            run_bundle_adjustment=False,
            drop_outlier_after_camera_merging=False,
            plot_reprojection_histograms=False,
        )
        merged = combine_results(None, (_child_result(reference), _child_result(other)), options=options).scene

        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertCountEqual(merged.get_valid_camera_indices(), [0, 1])
        self.assertEqual(merged.number_tracks(), 4)
        private_points = [
            track.point3()
            for track in merged.tracks()
            if track.numberMeasurements() == 1 and track.measurement(0)[0] == 1
        ]
        self.assertEqual(len(private_points), 1)
        np.testing.assert_allclose(private_points[0], np.array([3.0, 1.0, 4.0]), atol=1e-6)

    def test_missing_parent_stays_empty_for_internal_reconstruction(self) -> None:
        reference = _scene({0: (0.0, 0.0, 0.0)}, [_track((0.0, 0.0, 5.0), [(0, (10.0, 10.0))])])
        options = MergingOptions(reconstruct_internal_nodes=True, plot_reprojection_histograms=False)
        merged = combine_results(None, (_child_result(reference),), options=options).scene
        self.assertIsNone(merged)


if __name__ == "__main__":
    unittest.main()
