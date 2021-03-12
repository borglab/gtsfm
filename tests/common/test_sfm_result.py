"""Unit tests for the SfmResult class.

Authors: Ayush Baid
"""
import unittest

import gtsam
from gtsam import SfmData

import gtsfm.utils.io as io_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.sfm_result import SfmResult

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"
EXAMPLE_RESULT = SfmResult(io_utils.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE)), total_reproj_error=1.5e1,)

NULL_RESULT = SfmResult(SfmData(), total_reproj_error=float("Nan"))


class TestSfmResult(unittest.TestCase):
    """Unit tests for SfmResult"""

    def testEqualsWithSameObject(self):
        """Test equality function with the same object."""
        self.assertEqual(EXAMPLE_RESULT, EXAMPLE_RESULT)

    def testEqualsWithDifferentObject(self):
        """Test the equality function with different object, expecting false result."""
        other_example_file = "dubrovnik-1-1-pre.txt"
        other_result = SfmResult(
            io_utils.read_bal(gtsam.findExampleDataFile(other_example_file)), total_reproj_error=1.1e1,
        )

        self.assertNotEqual(EXAMPLE_RESULT, other_result)

    def testEqualsWithNullObject(self):
        """Tests equality of null object with itself and other valid object."""

        self.assertEqual(NULL_RESULT, NULL_RESULT)

        self.assertNotEqual(NULL_RESULT, EXAMPLE_RESULT)

    def testGetTrackLengthStatistics(self):
        """Test computation of mean and median track length."""
        expected_mean_length = 2.7142857142857144
        expected_median_length = 3.0

        mean_length, median_length, _ = EXAMPLE_RESULT.get_track_length_statistics()

        self.assertEqual(mean_length, expected_mean_length)
        self.assertEqual(median_length, expected_median_length)

    def test_filter_landmarks(self):
        """Tests filtering of SfmData based on reprojection error."""
        max_reproj_error = 15

        VALID_TRACK_INDICES = [0, 1, 5]

        # construct expected data w/ tracks with reprojection errors below the
        # threshold
        expected_data = GtsfmData(EXAMPLE_RESULT.gtsfm_data.number_images())
        for i in EXAMPLE_RESULT.gtsfm_data.get_valid_camera_indices():
            expected_data.add_camera(EXAMPLE_RESULT.gtsfm_data.get_camera(i), i)

        for j in VALID_TRACK_INDICES:
            expected_data.add_track(EXAMPLE_RESULT.gtsfm_data.get_track(j))

        # run the fn under test
        filtered_sfm_data = EXAMPLE_RESULT.filter_landmarks(max_reproj_error)

        # compare the SfmData objects
        self.assertEqual(filtered_sfm_data, expected_data)


if __name__ == "__main__":
    unittest.main()
