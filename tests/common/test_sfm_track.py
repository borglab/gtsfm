"""Unit tests for SfmTrack2d class.

Authors: Sushmita Warrier, Xiaolong Wu, John Lambert
"""

import numpy as np
from gtsam.utils.test_case import GtsamTestCase

from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d

SAMPLE_MEASUREMENTS = [
    SfmMeasurement(0, np.random.rand(2)),
    SfmMeasurement(2, np.random.rand(2)),
    SfmMeasurement(3, np.random.rand(2)),
    SfmMeasurement(5, np.random.rand(2)),
]


class TestSfmTrack2d(GtsamTestCase):
    def test_eq_check_with_same_measurements(self) -> None:
        """Tests the __eq__ function with the same set of measurements but with different ordering."""

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
        """Tests the __eq__ function with one track having subset of measurements of the other."""

        track_1 = SfmTrack2d(SAMPLE_MEASUREMENTS)
        # dropping the last measurement
        track_2 = SfmTrack2d(SAMPLE_MEASUREMENTS[:3])

        self.assertNotEqual(track_1, track_2)
        self.assertNotEqual(track_2, track_1)

    def test_eq_check_with_different_measurements(self) -> None:
        """Tests the __eq__ function with one measurement having different value of the 2d point."""

        track_1 = SfmTrack2d(SAMPLE_MEASUREMENTS)
        # changing the value of the last measurement
        old_measurement = SAMPLE_MEASUREMENTS[-1]
        track_2 = SfmTrack2d(SAMPLE_MEASUREMENTS[:3] + [SfmMeasurement(old_measurement.i, np.random.rand(2))])

        self.assertNotEqual(track_1, track_2)
        self.assertNotEqual(track_2, track_1)
