"""Unit tests for the two-view estimator.

Authors: Ayush Baid
"""
from gtsfm.common.keypoints import Keypoints
import unittest

import gtsam
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Pose3, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.io as io_utils
from gtsfm.two_view_estimator import TwoViewEstimator

GTSAM_EXAMPLE_FILE = "18pointExample1.txt"
EXAMPLE_DATA = io_utils.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))


class TestTwoViewEstimator(unittest.TestCase):
    """Unit tests for the 2-view estimator"""

    def test_bundle_adjust(self):
        """Tests the bundle adjustment for relative pose on a simulated scene."""

        # prepare the points from the measurement
        num_points = 5

        normalized_coordinates_i1 = []
        normalized_coordinates_i2 = []

        i1Ri2 = EXAMPLE_DATA.get_camera(1).pose().rotation()
        i1Ti2 = EXAMPLE_DATA.get_camera(1).pose().translation()
        i1Pi2 = Pose3(i1Ri2, i1Ti2)
        i2Pi1 = i1Pi2.inverse()

        i2Ei1 = EssentialMatrix(i2Pi1.rotation(), Unit3(i2Pi1.translation()))

        for i in range(num_points):
            track = EXAMPLE_DATA.get_track(i)
            pA = track.measurement(0)[1]
            pB = track.measurement(1)[1]

            normalized_coordinates_i1.append(pA)
            normalized_coordinates_i2.append(pB)

        normalized_coordinates_i1 = np.array(normalized_coordinates_i1)
        normalized_coordinates_i2 = np.array(normalized_coordinates_i2)

        print(normalized_coordinates_i1.shape)

        i2Ri1_optimized, i2Ui1_optimized = TwoViewEstimator.bundle_adjust(
            keypoints_i1=Keypoints(normalized_coordinates_i1),
            keypoints_i2=Keypoints(normalized_coordinates_i2),
            verified_corr_idxes=np.hstack([np.arange(normalized_coordinates_i1.shape[0]).reshape(-1, 1)] * 2),
            camera_intrinsics_i1=Cal3Bundler(),
            camera_intrinsics_i2=Cal3Bundler(),
            i2Ri1_initial=i2Ei1.rotation(),
            i2Ui1_initial=i2Ei1.direction(),
        )

        self.assertLessEqual(comp_utils.compute_relative_rotation_angle(i2Ri1_optimized, i2Ei1.rotation()), 1)
        self.assertLessEqual(comp_utils.compute_relative_unit_translation_angle(i2Ui1_optimized, i2Ei1.direction()), 1)

        # self.assertEqual(i2Ri1_optimized, i2Ei1.rotation())
        # self.assertEqual(i2Ui1_optimized, i2Ei1.direction())


if __name__ == "__main__":
    unittest.main()

