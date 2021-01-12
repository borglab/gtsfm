"""
Unit tests for the FeatureTrackGenerator class.

Authors: Sushmita Warrier, Xiaolong Wu, John Lambert
"""

import copy
from typing import Dict, List, Tuple

import numpy as np
from gtsam.utils.test_case import GtsamTestCase

from gtsfm.common.keypoints import Keypoints
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d

SAMPLE_MEASUREMENTS = [
    SfmMeasurement(0, np.random.rand(2)),
    SfmMeasurement(2, np.random.rand(2)),
    SfmMeasurement(3, np.random.rand(2)),
    SfmMeasurement(5, np.random.rand(2)),
]


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
    """Set up correspondences for each (i1,i2) pair.
    There should be 4 tracks, since we get one chained track
    as (i=0, k=0) -> (i=1, k=2) -> (i=2,k=3)
    """
    dummy_matches_dict = {
        (0, 1): np.array([[0, 2]]),
        (1, 2): np.array([[2, 3], [4, 5], [7, 9]]),
        (0, 2): np.array([[1, 8]]),
    }
    return dummy_matches_dict


class TestSfmTrack2d(GtsamTestCase):
    def test_eq_check_with_same_measurements(self) -> None:
        """Tests the __eq__ function with the same set of measurements but with
        different ordering."""

        # construct two tracks with different ordering of measurements
        track_1 = SfmTrack2d(SAMPLE_MEASUREMENTS)
        track_2 = SfmTrack2d(
            [
                SAMPLE_MEASUREMENTS[0],
                SAMPLE_MEASUREMENTS[3],
                SAMPLE_MEASUREMENTS[1],
                SAMPLE_MEASUREMENTS[2],
            ]
        )

        self.assertEqual(track_1, track_2)

    def test_eq_check_with_missing_measurements(self) -> None:
        """Tests the __eq__ function with one track having subset of
        measurements of the other.
        """

        track_1 = SfmTrack2d(SAMPLE_MEASUREMENTS)
        # dropping the last measurement
        track_2 = SfmTrack2d(SAMPLE_MEASUREMENTS[:3])

        self.assertNotEqual(track_1, track_2)
        self.assertNotEqual(track_2, track_1)

    def test_eq_check_with_different_measurements(self) -> None:
        """Tests the __eq__ function with one measurement having different value
        of the 2d point.
        """

        track_1 = SfmTrack2d(SAMPLE_MEASUREMENTS)
        # changing the value of the last measurement
        old_measurement = SAMPLE_MEASUREMENTS[-1]
        track_2 = SfmTrack2d(
            SAMPLE_MEASUREMENTS[:3]
            + [SfmMeasurement(old_measurement.i, np.random.rand(2))]
        )

        self.assertNotEqual(track_1, track_2)
        self.assertNotEqual(track_2, track_1)

    def test_generate_tracks_from_pairwise_matches_no_duplicates(self) -> None:
        """Tests that the tracks are being merged and mapped correctly."""
        dummy_keypoints_list = get_dummy_keypoints_list()
        dummy_matches_dict = get_dummy_matches()

        tracks = SfmTrack2d.generate_tracks_from_pairwise_matches(
            dummy_matches_dict, dummy_keypoints_list
        )
        # len(track) value for toy case strictly
        self.assertEqual(len(tracks), 4, "tracks incorrectly mapped")

    def test_generate_tracks_from_pairwise_matches_with_duplicates(
        self,
    ) -> None:
        """
        Tests that the tracks are being filtered correctly.
        Removes tracks that have two measurements in a single image.
        """
        dummy_keypoints_list = get_dummy_keypoints_list()

        dummy_matches_dict = get_dummy_matches()
        malformed_matches_dict = copy.deepcopy(dummy_matches_dict)

        # add erroneous correspondence
        malformed_matches_dict[(1, 1)] = np.array([[0, 3]])

        tracks = SfmTrack2d.generate_tracks_from_pairwise_matches(
            malformed_matches_dict, dummy_keypoints_list
        )

        # check that the length of the observation list corresponding to each key
        # is the same. Only good tracks will remain
        self.assertEqual(len(tracks), 4, "Tracks not filtered correctly")
