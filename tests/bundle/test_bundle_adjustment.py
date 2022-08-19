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

    def setUp(self):
        super().setUp()

        output_reproj_error_thresh = 100
        self.ba = BundleAdjustmentOptimizer(output_reproj_error_thresh)

        self.test_data = EXAMPLE_DATA

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


if __name__ == "__main__":
    unittest.main()
