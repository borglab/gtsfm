"""Unit tests for the two-view estimator.

Authors: Ayush Baid
"""
import unittest
from unittest.mock import ANY, MagicMock, patch

import gtsam
import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3, Unit3
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.metrics import GtsfmMetricsGroup

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.io as io_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.two_view_estimator import TwoViewEstimator


GTSAM_EXAMPLE_FILE = "5pointExample1.txt"
EXAMPLE_DATA = io_utils.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))

# dummy data used for mocks
MOCK_DA_OUTPUT = MagicMock()
MOCK_DA_METRICS = MagicMock()
MOCK_BA_OUTPUT = MagicMock()
MOCK_BA_METRICS = MagicMock()


class TestTwoViewEstimator(unittest.TestCase):
    """Unit tests for the 2-view estimator.

    Uses GTSAM's 5-point example for ground truth tracks and cameras. See `gtsam/examples/Data/5pointExample1.txt` for
    details.
    """

    @patch("gtsfm.data_association.data_assoc.DataAssociation.run", return_value=[MOCK_DA_OUTPUT, MOCK_DA_METRICS])
    @patch(
        "gtsfm.bundle.bundle_adjustment.BundleAdjustmentOptimizer.run", return_value=[MOCK_BA_OUTPUT, MOCK_BA_METRICS]
    )
    def test_bundle_adjust(self, mock_ba_run: MagicMock, mock_da_run: MagicMock):
        """Tests 2-view BA using mocked data to confirm calls to data-association and bundle-adjustment modules"""
        # Extract example poses.
        camera_i1 = PinholeCameraCal3Bundler()
        camera_i2 = PinholeCameraCal3Bundler(Pose3(Rot3.RzRyRx(0, np.pi / 2, 0), np.array([1, 0, 0])), Cal3Bundler())
        i2Ti1 = camera_i2.pose().between(camera_i1.pose())

        keypoints_i1 = MagicMock()
        keypoints_i2 = MagicMock()
        verified_corr_idxs = MagicMock()

        ba_output_poses = MagicMock(return_value=[camera_i1.pose(), camera_i2.pose()])
        MOCK_BA_OUTPUT.get_camera_poses = ba_output_poses

        # Perform bundle adjustment.
        _, _, _ = TwoViewEstimator.bundle_adjust(
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            verified_corr_idxs=verified_corr_idxs,
            camera_intrinsics_i1=camera_i1.calibration(),
            camera_intrinsics_i2=camera_i2.calibration(),
            i2Ri1_initial=i2Ti1.rotation(),
            i2Ui1_initial=Unit3(i2Ti1.translation()),
        )

        mock_da_run.assert_called_once_with(
            num_images=2,
            cameras={
                0: ANY,
                1: ANY,
            },  # cannot check exact camera values right now because the __eq__ function is broken
            corr_idxs_dict={(0, 1): verified_corr_idxs},
            keypoints_list=[keypoints_i1, keypoints_i2],
        )
        mock_ba_run.assert_called_once_with(MOCK_DA_OUTPUT)


if __name__ == "__main__":
    unittest.main()
