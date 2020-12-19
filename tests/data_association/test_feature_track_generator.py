
"""
Unit tests for the FeatureTrackGenerator class.

Authors: Sushmita Warrier, Xiaolong Wu, John Lambert
"""

import copy
from typing import Dict, List, Tuple

import numpy as np
from gtsam.utils.test_case import GtsamTestCase

from common.keypoints import Keypoints
from data_association.feature_tracks import FeatureTrackGenerator


def get_dummy_keypoints_list() -> List[Keypoints]:
    """ """
    img1_kp_coords = np.array([[1,1], [2,2], [3,3]])
    img1_kp_scale = np.array([6.0, 9.0, 8.5])
    img2_kp_coords = np.array(
        [
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6],
            [7, 7],
            [8, 8],
        ]
    )
    img3_kp_coords = np.array(
        [
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6],
            [7, 7],
            [8, 8],
            [9, 9],
            [10,10],
        ]
    )
    keypoints_list = [
        Keypoints(coordinates=img1_kp_coords, scale=img1_kp_scale),
        Keypoints(coordinates=img2_kp_coords),
        Keypoints(coordinates=img3_kp_coords),
    ]
    return keypoints_list


def get_dummy_matches() -> Dict[Tuple[int,int], np.ndarray]:
    """ Set up correspondences for each (i1,i2) pair.
    There should be 4 tracks, since we get one chained track 
    as (i=0, k=0) -> (i=1, k=2) -> (i=2,k=3)
    """
    dummy_matches_dict = {
        (0, 1): np.array([[0, 2]]),
        (1, 2): np.array([[2, 3], [4, 5], [7, 9]]),
        (0, 2): np.array([[1, 8]]),
    }
    return dummy_matches_dict


class TestFeatureTrackGenerator(GtsamTestCase):

    def setUp(self):
        """
        Set up the data association module.
        """
        super().setUp()

    def test_track_generation(self) -> None:
        """
        Tests that the tracks are being merged and mapped correctly from the dummy matches.
        """
        dummy_keypoints_list = get_dummy_keypoints_list()
        dummy_matches_dict = get_dummy_matches()

        tracks = FeatureTrackGenerator(dummy_matches_dict, dummy_keypoints_list)
        # len(track) value for toy case strictly
        assert len(tracks.filtered_landmark_data) == 4, "tracks incorrectly mapped"


    def test_erroneous_track_filtering(self) -> None:
        """
        Tests that the tracks are being filtered correctly.
        Removes tracks that have two measurements in a single image.
        """
        dummy_keypoints_list = get_dummy_keypoints_list()

        dummy_matches_dict = get_dummy_matches()
        malformed_matches_dict = copy.deepcopy(dummy_matches_dict)

        # add erroneous correspondence
        malformed_matches_dict[(1, 1)] = np.array([[0, 3]])

        tracks = FeatureTrackGenerator(malformed_matches_dict, dummy_keypoints_list)

        # check that the length of the observation list corresponding to each key
        # is the same. Only good tracks will remain
        assert len(tracks.filtered_landmark_data) == 4, "Tracks not filtered correctly"

