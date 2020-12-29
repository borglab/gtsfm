"""Unit tests for bundle adjustment.

Authors: Ayush Baid
"""

import unittest

import dask
import gtsam
import numpy as np

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.sfm_result import SfmData
from gtsfm.data_association.feature_tracks import SfmMeasurement, SfmTrack

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"


def read_example_data() -> SfmData:
    """Read the example data from GTSAM

    Returns:
        SfmData corresponding to GTSAM_EXAMPLE_FILE.
    """
    gtsam_data = gtsam.readBal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))

    data = SfmData()

    # add cameras
    for idx in range(gtsam_data.number_cameras()):
        data.add_camera(idx, gtsam_data.camera(idx))

    # add tracks
    for idx in range(gtsam_data.number_tracks()):
        gtsam_track = gtsam_data.track(idx)

        landmark = gtsam_track.point3()

        measurements = []
        for measurement_idx in range(gtsam_track.number_measurements()):
            i, uv = gtsam_track.measurement(measurement_idx)

            measurements.append(SfmMeasurement(i, uv))

        data.add_track(SfmTrack(measurements, landmark))

    return data


class TestBundleAdjustmentOptimizer(unittest.TestCase):
    """Unit tests for BundleAdjustmentOptimizer class."""

    def setUp(self):
        super().setUp()

        self.obj = BundleAdjustmentOptimizer()

        self.test_data = read_example_data()

    def test_simple_scene(self):
        """Test the simple scene using the `run` API."""

        computed_result = self.obj.run(self.test_data)

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

        expected_result = self.obj.run(self.test_data)

        computed_result = self.obj.create_computation_graph(
            dask.delayed(sfm_data_graph)
        )

        with dask.config.set(scheduler="single-threaded"):
            result = dask.compute(computed_result)[0]

        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
