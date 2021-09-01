"""Unit tests for bundle adjustment.

Authors: Ayush Baid
"""

import unittest

import dask
import gtsam

import gtsfm.utils.io as io_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"
EXAMPLE_DATA = io_utils.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))


class TestBundleAdjustmentOptimizer(unittest.TestCase):
    """Unit tests for BundleAdjustmentOptimizer class."""

    def test_create_computation_graph(self):
        """Test the simple scene as dask computation graph."""
        test_obj = BundleAdjustmentOptimizer(shared_calib=True, output_reproj_error_thresh=3)

        sfm_data_graph = dask.delayed(EXAMPLE_DATA)

        expected_result, _ = test_obj.run(EXAMPLE_DATA)

        computed_result, _ = test_obj.create_computation_graph(dask.delayed(sfm_data_graph))

        with dask.config.set(scheduler="single-threaded"):
            result = dask.compute(computed_result)[0]

        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
