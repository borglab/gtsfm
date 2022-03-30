"""Tests for CorrespondenceGraph class.

Authors: Travis Driver
"""

import unittest
from typing import List, Dict, Tuple

import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.data_association.correspondence_graph import CorrespondenceGraph


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
    keypoints_list = [
        Keypoints(coordinates=img1_kp_coords, scales=img1_kp_scale),
        Keypoints(coordinates=img2_kp_coords),
        Keypoints(coordinates=img3_kp_coords),
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


class TestCorrespondenceGraph(unittest.TestCase):
    """"""

    def setUp(self):
        """Set up the data association module."""
        super().setUp()

        self.corr_graph = CorrespondenceGraph(get_dummy_matches(), get_dummy_keypoints_list())

    def test_get_aggregate_assoc_matrix(self) -> None:
        """Test converting a CorrespondenceGraph to an aggregate association matrix.

        Consider a universe with three items A, B, C. Let view 1 see (A, B), view 2 see (B, C), and view 3 see (A, C).
        Then the aggregate association matrix P is

        P =
        [
                 1     2     3
                 A  B  B  C  A  C
            1 A [1, 0, 0, 0, 1, 0],
              B [0, 1, 1, 0, 0, 0],
            2 B [0, 1, 1, 0, 0, 0],
              C [0, 0, 0, 1, 0, 1],
            3 A [1, 0, 0, 0, 1, 0],
              C [0, 0, 0, 1, 0, 1]
        ]

        """
        P_expected = np.array(
            [
                [1, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1],
            ]
        )
        P_computed = self.corr_graph.get_aggregate_assoc_matrix()
        np.testing.assert_allclose(P_expected, P_computed)
