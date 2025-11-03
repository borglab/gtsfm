"""Unit tests for bundle adjustment.

Authors: Ayush Baid
"""

import unittest

import dask
import gtsam  # type: ignore
import numpy as np

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"
EXAMPLE_DATA = GtsfmData.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))


class TestBundleAdjustmentOptimizer(unittest.TestCase):
    """Unit tests for BundleAdjustmentOptimizer class."""

    def setUp(self):
        super().setUp()

        reproj_error_thresholds = [100.0]
        self.ba = BundleAdjustmentOptimizer(reproj_error_thresholds=reproj_error_thresholds)

        self.test_data = EXAMPLE_DATA

    def _clone_with_shared_calibration(self, data: GtsfmData) -> GtsfmData:
        """Return a copy of ``data`` where all cameras share the first camera's calibration."""
        shared_clone = GtsfmData(data.number_images())
        valid_camera_indices = data.get_valid_camera_indices()
        assert len(valid_camera_indices) > 0
        first_cam = data.get_camera(valid_camera_indices[0])
        assert first_cam is not None
        camera_type = type(first_cam)
        shared_calibration = first_cam.calibration()

        for idx in valid_camera_indices:
            cam = data.get_camera(idx)
            assert cam is not None
            shared_clone.add_camera(idx, camera_type(cam.pose(), shared_calibration))

        for track_idx in range(data.number_tracks()):
            shared_clone.add_track(data.get_track(track_idx))

        return shared_clone

    # def test_simple_scene(self):
    #     """Test the simple scene using the `run_ba` API."""

    #     computed_result = self.ba.run_ba(self.test_data)

    #     expected_error = 0.046137573704557046

    #     self.assertTrue(np.isclose(expected_error, computed_result.total_reproj_error, atol=1e-2, rtol=1e-2))

    def test_create_computation_graph(self):
        """Test the simple scene as dask computation graph."""
        sfm_data_graph = dask.delayed(self.test_data)

        absolute_pose_priors = [None] * EXAMPLE_DATA.number_images()
        relative_pose_priors = {}

        expected_result, _, _ = self.ba.run_ba(
            self.test_data, absolute_pose_priors=absolute_pose_priors, relative_pose_priors=relative_pose_priors
        )

        computed_result, _ = self.ba.create_computation_graph(
            sfm_data_graph,
            absolute_pose_priors,
            relative_pose_priors,
            cameras_gt=[None] * self.test_data.number_images(),
        )

        with dask.config.set(scheduler="single-threaded"):
            result = dask.compute(computed_result)[0]

        self.assertEqual(result, expected_result)

    def test_values_roundtrip_without_shared_calib(self):
        """Ensure GtsfmData <-> gtsam.Values conversion preserves scene when calibrations are independent."""
        values = self.test_data.to_values(shared_calib=False)
        reconstructed = GtsfmData.from_values(values, initial_data=self.test_data, shared_calib=False)
        self.assertEqual(reconstructed, self.test_data)

    def test_values_roundtrip_with_shared_calib(self):
        """Ensure GtsfmData <-> gtsam.Values conversion preserves scene when calibration is shared."""
        shared_calib_data = self._clone_with_shared_calibration(self.test_data)
        values = shared_calib_data.to_values(shared_calib=True)
        reconstructed = GtsfmData.from_values(values, initial_data=shared_calib_data, shared_calib=True)
        self.assertEqual(reconstructed, shared_calib_data)

    def test_from_values_without_initial_data(self):
        """Ensure from_values succeeds without access to the original initial_data."""
        values = self.test_data.to_values(shared_calib=False)
        reconstructed = GtsfmData.from_values(values)

        self.assertEqual(reconstructed.number_images(), self.test_data.number_images())
        self.assertEqual(reconstructed.get_valid_camera_indices(), self.test_data.get_valid_camera_indices())
        self.assertEqual(reconstructed.number_tracks(), self.test_data.number_tracks())

        for camera_idx in self.test_data.get_valid_camera_indices():
            original = self.test_data.get_camera(camera_idx)
            rebuilt = reconstructed.get_camera(camera_idx)
            self.assertIsNotNone(original)
            self.assertIsNotNone(rebuilt)
            assert original is not None and rebuilt is not None
            self.assertTrue(original.pose().equals(rebuilt.pose(), 1e-9))
            self.assertTrue(original.calibration().equals(rebuilt.calibration(), 1e-9))

        for track_idx in range(self.test_data.number_tracks()):
            original_point = self.test_data.get_track(track_idx).point3()
            rebuilt_point = reconstructed.get_track(track_idx).point3()
            self.assertTrue(np.allclose(rebuilt_point, original_point))

    def test_run_simple_ba(self):
        """Test run_simple_ba on simple scene."""
        computed_result, error = self.ba.run_simple_ba(self.test_data)

        self.assertEqual(computed_result.number_images(), self.test_data.number_images())
        self.assertAlmostEqual(error, 2.32, places=2)


if __name__ == "__main__":
    unittest.main()
