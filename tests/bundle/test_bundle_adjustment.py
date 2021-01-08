"""Unit tests for bundle adjustment.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path

import dask
import gtsam
import numpy as np
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer

# Simple example from GTSAM
GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"
SIMPLE_DATA = gtsam.readBal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))

# Door scene from the Lund Dataset
DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
DOOR_DATASET = DATA_ROOT_PATH / "door_initialization.bal"
DOOR_DATA = gtsam.readBal(str(DOOR_DATASET))


class TestBundleAdjustmentOptimizer(unittest.TestCase):
    """Unit tests for BundleAdjustmentOptimizer class."""

    def test_simple_scene_with_nonrobust_noise(self):
        """Test a simple scene with simple isotropic measurement noise."""
        obj = BundleAdjustmentOptimizer(use_robust_measurement_noise=False)

        computed_result = obj.run(SIMPLE_DATA)

        expected_error = 0.046137573704557046

        np.testing.assert_allclose(
            expected_error,
            computed_result.total_reproj_error,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_simple_scene_with_robust_noise(self):
        """Tests a simple scene with robust measurement noise."""
        obj = BundleAdjustmentOptimizer(use_robust_measurement_noise=True)

        computed_result = obj.run(SIMPLE_DATA)

        expected_error = 0.046137573704557046

        np.testing.assert_allclose(
            expected_error,
            computed_result.total_reproj_error,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_door_scene_with_nonrobust_noise(self):
        """Tests the door scene with simple isotropic measurement noise."""
        obj = BundleAdjustmentOptimizer(use_robust_measurement_noise=False)

        computed_result = obj.run(DOOR_DATA)

        expected_error = 452.25703368662914

        np.testing.assert_allclose(
            expected_error,
            computed_result.total_reproj_error,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_door_scene_with_robust_noise(self):
        """Tests the door scene with robust measurement noise."""
        obj = BundleAdjustmentOptimizer(use_robust_measurement_noise=True)

        computed_result = obj.run(DOOR_DATA)

        expected_error = 331.1219617185047

        np.testing.assert_allclose(
            expected_error,
            computed_result.total_reproj_error,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_create_computation_graph(self):
        """Test the simple scene as dask computation graph."""
        sfm_data_graph = dask.delayed(SIMPLE_DATA)

        obj = BundleAdjustmentOptimizer(use_robust_measurement_noise=False)

        expected_result = obj.run(SIMPLE_DATA)

        computed_result = obj.create_computation_graph(
            dask.delayed(sfm_data_graph)
        )

        with dask.config.set(scheduler="single-threaded"):
            result = dask.compute(computed_result)[0]

        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
