"""Unit tests for 2-step bundle adjustment.

Authors: Ayush Baid
"""
import unittest

import dask
import gtsam
import numpy as np

import gtsfm.utils.io as io_utils
from gtsfm.bundle.two_step_ba import TwoStepBA
from gtsfm.common.gtsfm_data import GtsfmData

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"
EXAMPLE_DATA = io_utils.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))


class TestTwoStepBA(unittest.TestCase):
    """Unit tests for TwoStepBA class."""

    def setUp(self):
        super().setUp()

        output_reproj_error_thresh = 0.3
        intermediate_reproj_error_thresh = 1
        self.ba = TwoStepBA(
            intermediate_reproj_error_thresh=intermediate_reproj_error_thresh,
            output_reproj_error_thresh=output_reproj_error_thresh,
        )

        self.test_data: GtsfmData = EXAMPLE_DATA
        # outlier_track = EXAMPLE_DATA.get_track(1)

        # self.test_data.add_track()

    def test_simple_scene(self):
        """Test the simple scene using the `run` API."""

        _, computed_result, boolean_mask = self.ba.run(
            self.test_data, absolute_pose_priors=None, relative_pose_priors={}, verbose=True
        )

        expected_error = 0.058017
        expected_num_tracks = 6

        np.testing.assert_allclose(
            expected_error, computed_result.get_avg_scene_reprojection_error(), atol=1e-2, rtol=1e-2
        )
        self.assertEqual(expected_num_tracks, computed_result.number_tracks())
        self.assertListEqual([True, True, True, True, True, True, False], boolean_mask)

    def test_create_computation_graph(self):
        """Test the simple scene as dask computation graph."""
        sfm_data_graph = dask.delayed(self.test_data)

        absolute_pose_priors = [None] * EXAMPLE_DATA.number_images()
        relative_pose_priors = {}

        expected_result, _, _ = self.ba.run(
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
