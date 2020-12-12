"""Unit tests for bundle adjustment.

Authors: Ayush Baid
"""

import unittest

import dask
import gtsam
import numpy as np

from bundle.bundle_adjustment import BundleAdjustmentOptimizer

GTSAM_EXAMPLE_FILE = 'dubrovnik-3-7-pre'
TEST_SFM_DATA = gtsam.readBal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))


class TestBundleAdjustmentOptimizer(unittest.TestCase):
    """Unit tests for BundleAdjustmentOptimizer class."""

    def setUp(self):
        super().setUp()

        self.obj = BundleAdjustmentOptimizer()

    def test_simple_scene(self):
        """Test the simple scene using the `run` API."""

        computed_result = self.obj.run(TEST_SFM_DATA)

        expected_error = 0.046137573704557046
        self.assertTrue(np.isclose(
            expected_error, computed_result.total_reproj_error))

    def test_create_computation_graph(self):
        """Test the simple scene as dask computation graph."""
        sfm_data_graph = dask.delayed(TEST_SFM_DATA)

        expected_result = self.obj.run(TEST_SFM_DATA)

        computed_result = self.obj.create_computation_graph(
            dask.delayed(sfm_data_graph)
        )

        with dask.config.set(scheduler='single-threaded'):
            result = dask.compute(computed_result)[0]

        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
