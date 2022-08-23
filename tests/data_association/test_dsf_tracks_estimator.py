"""Unit tests for the DsfTracksEstimator class.

Authors: Sushmita Warrier, Xiaolong Wu, John Lambert, Travis Driver
"""

import copy
from typing import Dict, List, Tuple

import numpy as np
from gtsam.utils.test_case import GtsamTestCase

from gtsfm.common.keypoints import Keypoints
from gtsfm.data_association.dsf_tracks_estimator import DsfTracksEstimator


class TestDsfTracksEstimator(GtsamTestCase):
    """ """

    def setUp(self):
        self.estimator = DsfTracksEstimator()

    def test_generate_tracks_from_pairwise_matches_no_duplicates(self) -> None:
        """Tests that the tracks are being merged and mapped correctly."""
        dummy_keypoints_list = get_dummy_keypoints_list()
        dummy_matches_dict = get_dummy_matches()

        tracks = self.estimator.run(dummy_matches_dict, dummy_keypoints_list)
        self.assertEqual(len(tracks), 4, "Tracks not filtered correctly")

    def test_generate_tracks_from_pairwise_matches_with_duplicates(
        self,
    ) -> None:
        """Tests that the tracks are being filtered correctly. Removes tracks that have two measurements in a single
        image.
        """
        dummy_keypoints_list = get_dummy_keypoints_list()

        dummy_matches_dict = get_dummy_matches()
        malformed_matches_dict = copy.deepcopy(dummy_matches_dict)

        # add erroneous correspondence
        malformed_matches_dict[(1, 1)] = np.array([[0, 3]])

        tracks = self.estimator.run(malformed_matches_dict, dummy_keypoints_list)

        # check that the length of the observation list corresponding to each key
        # is the same. Only good tracks will remain
        self.assertEqual(len(tracks), 4, "Tracks not filtered correctly")

    def test_generate_tracks_from_pairwise_matches_nontransitive(
        self,
    ) -> None:
        """Tests DSF for non-transitive matches.

        Test will result in no tracks since nontransitive tracks are naively discarded by DSF.
        """
        dummy_keypoints_list = get_dummy_keypoints_list()
        nontransitive_matches_dict = get_nontransitive_matches()  # contains one non-transitive track
        tracks = self.estimator.run(nontransitive_matches_dict, dummy_keypoints_list)
        print(tracks)
        self.assertEqual(len(tracks), 0, "Tracks not filtered correctly")


def get_dummy_keypoints_list() -> List[Keypoints]:
    """ """
    img1_kp_coords = np.array([[1, 1], [2, 2], [3, 3]])
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
            [10, 10],
        ]
    )
    img4_kp_coords = np.array(
        [
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
        ]
    )
    keypoints_list = [
        Keypoints(coordinates=img1_kp_coords, scales=img1_kp_scale),
        Keypoints(coordinates=img2_kp_coords),
        Keypoints(coordinates=img3_kp_coords),
        Keypoints(coordinates=img4_kp_coords),
    ]
    return keypoints_list


def get_dummy_matches() -> Dict[Tuple[int, int], np.ndarray]:
    """Set up correspondences for each (i1,i2) pair. There should be 4 tracks, since we get one chained track as
    (i=0, k=0) -> (i=1, k=2) -> (i=2,k=3).
    """
    dummy_matches_dict = {
        (0, 1): np.array([[0, 2]]),
        (1, 2): np.array([[2, 3], [4, 5], [7, 9]]),
        (0, 2): np.array([[1, 8]]),
    }
    return dummy_matches_dict


def get_nontransitive_matches() -> Dict[Tuple[int, int], np.ndarray]:
    """Set up correspondences for each (i1,i2) pair that violates transitivity.
    
    (i=0, k=0)
         |    \
         |     \
    (i=1, k=2)--(i=2,k=3)--(i=3, k=4)--(i=1, k=1)

    Transitivity is violated due to the match between frames 0 and 3. 
    """
    nontransitive_matches_dict = {
        (0, 1): np.array([[0, 2]]),
        (1, 2): np.array([[2, 3]]),
        (0, 2): np.array([[0, 3]]),
        (0, 3): np.array([[1, 4]]),
        (2, 3): np.array([[3, 4]]),
    }
    return nontransitive_matches_dict
