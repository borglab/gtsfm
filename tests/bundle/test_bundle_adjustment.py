
import gtsam

import numpy as np

import dask

import unittest

from bundle.bundle_adjustment import BundleAdjustmentBase


class TestBundleAdjustment(unittest.TestCase):
    """Main tests for bundle adjustment base class"""

    def setUp(self):
        super(TestBundleAdjustment, self).setUp()

        self.obj = BundleAdjustmentBase()

    def test_bundle_adjustment(self):
        """Test the dask bundle adjustment pipline"""

        input_file_name = "dubrovnik-3-7-pre"
        input_file = gtsam.findExampleDataFile(input_file_name)

        # Load the SfM data from file
        scene_data = gtsam.readBal(input_file)

        computed_result = self.obj.run(scene_data)

        expected_error = 0.046137573704557046
        self.assertTrue(np.isclose(expected_error, computed_result.get_error()))

    def test_create_computation_graph(self):

        input_file_name = "dubrovnik-3-7-pre"
        input_file = gtsam.findExampleDataFile(input_file_name)

        # Load the SfM data from file
        scene_data = gtsam.readBal(input_file)

        computed_result = self.obj.create_computation_graph(
            dask.delayed(scene_data)
        )

        expected_result = self.obj.run(scene_data)

        with dask.config.set(scheduler='single-threaded'):
            dask_results = dask.compute(computed_result)[0]

        self.assertEqual(dask_results.get_error(), expected_result.get_error())


if __name__ == '__main__':
    unittest.main()
