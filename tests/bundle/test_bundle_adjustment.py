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

    def test_create_computation_graph(self):
        """Test the simple scene as dask computation graph."""
        sfm_data_graph = dask.delayed(self.test_data)

        expected_result, _ = self.obj.run(self.test_data)

        computed_result, _ = self.obj.create_computation_graph(dask.delayed(sfm_data_graph))

        graph = test_obj.create_computation_graph(dask.delayed(sfm_data_graph))
        with dask.config.set(scheduler="single-threaded"):
            result = dask.compute(graph)[0]

        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
