"""Unit tests for the CppDsfTracksEstimator class."""

import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.data_association.cpp_dsf_tracks_estimator import CppDsfTracksEstimator
from tests.data_association.test_dsf_tracks_estimator import TestDsfTracksEstimator


class TestCppDsfTracksEstimator(TestDsfTracksEstimator):
    """ """

    def setUp(self) -> None:
        self.estimator = CppDsfTracksEstimator()

    def test_nonrank2_keypoint_coordinates_raises(self) -> None:

        matches_dict = {(0, 1): np.random.randint(low=0, high=500, size=(100, 2), dtype=np.int32)}
        keypoints_list = [Keypoints(np.zeros((0, 2))), Keypoints(np.zeros((0,)))]

        with self.assertRaises(ValueError):
            estimator.run(matches_dict=matches_dict, keypoints_list=keypoints_list)
