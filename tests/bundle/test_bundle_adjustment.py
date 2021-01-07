"""Unit tests for bundle adjustment.

Authors: Ayush Baid
"""

import unittest

import dask
import gtsam
import numpy as np
from gtsam import SfmData, SfmTrack

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"
EXAMPLE_DATA = gtsam.readBal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))


class TestBundleAdjustmentOptimizer(unittest.TestCase):
    """Unit tests for BundleAdjustmentOptimizer class."""

    def setUp(self):
        super().setUp()

        self.test_data = EXAMPLE_DATA

    def test_simple_scene_with_nonrobust_noise(self):
        """Test a simple scene with simple isotropic measurement noise."""
        obj = BundleAdjustmentOptimizer(use_robust_measurement_noise=False)

        computed_result = obj.run(self.test_data)

        expected_error = 0.046137573704557046

        self.assertTrue(
            np.isclose(
                expected_error,
                computed_result.total_reproj_error,
                atol=1e-2,
                rtol=1e-2,
            )
        )

    def test_simple_scene_with_robust_noise(self):
        """Tests a simple scene with robust measurement noise."""
        obj = BundleAdjustmentOptimizer(use_robust_measurement_noise=True)

        computed_result = obj.run(self.test_data)

        expected_error = 0.046137573704557046

        self.assertTrue(
            np.isclose(
                expected_error,
                computed_result.total_reproj_error,
                atol=1e-2,
                rtol=1e-2,
            )
        )

    def test_create_computation_graph(self):
        """Test the simple scene as dask computation graph."""
        sfm_data_graph = dask.delayed(self.test_data)

        obj = BundleAdjustmentOptimizer(use_robust_measurement_noise=False)

        expected_result = obj.run(self.test_data)

        computed_result = obj.create_computation_graph(
            dask.delayed(sfm_data_graph)
        )

        with dask.config.set(scheduler="single-threaded"):
            result = dask.compute(computed_result)[0]

        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
