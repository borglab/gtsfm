"""Unit tests for bundle adjustment.

Authors: Ayush Baid
"""

import unittest

import dask
import gtsam
import numpy as np

import gtsfm.utils.io as io_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"
EXAMPLE_DATA = io_utils.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))


class TestBundleAdjustmentOptimizer(unittest.TestCase):
    """Unit tests for BundleAdjustmentOptimizer class."""

    def setUp(self):
        super().setUp()

        self.obj = BundleAdjustmentOptimizer()

        self.test_data = EXAMPLE_DATA

    # def test_simple_scene(self):
    #     """Test the simple scene using the `run` API."""

    #     computed_result = self.obj.run(self.test_data)

    #     expected_error = 0.046137573704557046

    #     self.assertTrue(np.isclose(expected_error, computed_result.total_reproj_error, atol=1e-2, rtol=1e-2))

    def test_create_computation_graph(self):
        """Test the simple scene as dask computation graph."""
        sfm_data_graph = dask.delayed(self.test_data)

        expected_result = self.obj.run(self.test_data)

        computed_result = self.obj.create_computation_graph(dask.delayed(sfm_data_graph))

        with dask.config.set(scheduler="single-threaded"):
            result = dask.compute(computed_result)[0]

        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
