"""Unit tests for bundle adjustment.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path

import dask
import gtsam
import numpy as np

import gtsfm.utils.io as io_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData

# Simple example from GTSAM
GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"
EXAMPLE_DATA = io_utils.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))


class TestBundleAdjustmentOptimizer(unittest.TestCase):
    """Unit tests for BundleAdjustmentOptimizer class."""

    def test_simple_scene_with_individual_calibration(self):
        """Test the simple scene w/ individual calibration for each camera using the `run` API."""
        expected_mean_track_length = 2.7142857142857144
        expected_median_track_length = 3.0

        output_reproj_error_thresh = 3
        test_obj = BundleAdjustmentOptimizer(output_reproj_error_thresh, shared_calib=False)

        computed_result = test_obj.run(EXAMPLE_DATA)

        (mean_track_length, median_track_length,) = computed_result.get_track_length_statistics()

        np.testing.assert_allclose(
            mean_track_length, expected_mean_track_length, atol=1e-2, rtol=1e-2,
        )

        np.testing.assert_allclose(
            median_track_length, expected_median_track_length, atol=1e-2, rtol=1e-2,
        )

    def test_simple_scene_with_shared_calibration(self):
        """Test the simple scene w/ shared calibration across all camera using the `run` API.

        Note: higher error expected because intrinsics are different for
        cameras in the example data.
        """
        expected_mean_track_length = 2.7142857142857144
        expected_median_track_length = 3.0

        output_reproj_error_thresh = 3
        test_obj = BundleAdjustmentOptimizer(output_reproj_error_thresh, shared_calib=True)

        computed_result = test_obj.run(EXAMPLE_DATA)
        (mean_track_length, median_track_length,) = computed_result.get_track_length_statistics()

        np.testing.assert_allclose(
            mean_track_length, expected_mean_track_length, atol=1e-2, rtol=1e-2,
        )

        np.testing.assert_allclose(
            median_track_length, expected_median_track_length, atol=1e-2, rtol=1e-2,
        )

    def test_create_computation_graph(self):
        """Test the simple scene as dask computation graph."""
        output_reproj_error_thresh = 3
        test_obj = BundleAdjustmentOptimizer(output_reproj_error_thresh, shared_calib=False)

        sfm_data_graph = dask.delayed(EXAMPLE_DATA)
        expected_result = test_obj.run(EXAMPLE_DATA)

        graph = test_obj.create_computation_graph(dask.delayed(sfm_data_graph))
        with dask.config.set(scheduler="single-threaded"):
            result = dask.compute(graph)[0]

        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
