"""Unit tests for comparison functions.

Authors: Ayush Baid
"""
import unittest

import numpy as np
from gtsam import Cal3_S2, Pose3, Rot3
from gtsam.examples import SFMdata

import utils.geometry_comparisons as geometry_comparisons

POSE_LIST = SFMdata.createPoses(Cal3_S2())


class TestComparisons(unittest.TestCase):
    """Unit tests for comparison functions."""

    def test_compare_poses_exact(self):
        self.assertTrue(
            geometry_comparisons.compare_global_poses(POSE_LIST, POSE_LIST)
        )

    def test_compare_poses_with_scaled_translations(self):
        scale_factor = 1.2
        pose_list_ = [Pose3(x.rotation(),
                            x.translation() * scale_factor) for x in POSE_LIST]

        self.assertTrue(geometry_comparisons.compare_global_poses(
            POSE_LIST, pose_list_))

    def test_compare_poses_with_origin_shift(self):
        new_origin = Pose3(
            Rot3.RzRyRx(0.3, 0.1, -0.27),
            np.array([-20.0, +19.0, 3.5]))

        pose_list_ = [new_origin.between(x) for x in POSE_LIST]

        self.assertTrue(geometry_comparisons.compare_global_poses(
            POSE_LIST, pose_list_))

    def test_compare_poses_with_different_ordering(self):

        pose_list = [POSE_LIST[1], POSE_LIST[2], POSE_LIST[3]]
        pose_list_ = [POSE_LIST[2], POSE_LIST[3], POSE_LIST[1]]

        self.assertTrue(geometry_comparisons.compare_global_poses(
            pose_list, pose_list_))


if __name__ == '__main__':
    unittest.main()
