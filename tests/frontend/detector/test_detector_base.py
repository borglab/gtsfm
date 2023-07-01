"""Tests for frontend's base detector class.

Authors: Ayush Baid
"""
import pickle
import unittest
from pathlib import Path

import numpy as np

from gtsfm.frontend.detector.dummy_detector import DummyDetector
from gtsfm.loader.olsson_loader import OlssonLoader

# defining the path for test data
DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent / "data"
TEST_DATA_PATH = DATA_ROOT_PATH / "set1_lund_door"


class TestDetectorBase(unittest.TestCase):
    """Main test class for detector base class in frontend."""

    def setUp(self):
        super().setUp()
        self.detector = DummyDetector()
        self.loader = OlssonLoader(TEST_DATA_PATH, image_extension="JPG")

    def test_number_of_detections(self):
        """Tests that the number of detections is less than the maximum number configured."""
        test_image = self.loader.get_image(0)
        keypoints = self.detector.detect(test_image)

        self.assertLessEqual(len(keypoints), self.detector.max_keypoints)

    def test_coordinates_range(self):
        """Tests that each coordinate is within the image bounds."""
        test_image = self.loader.get_image(0)
        keypoints = self.detector.detect(test_image)

        np.testing.assert_array_equal(keypoints.coordinates[:, 0] >= 0, True)
        np.testing.assert_array_equal(keypoints.coordinates[:, 0] <= test_image.width, True)
        np.testing.assert_array_equal(keypoints.coordinates[:, 1] >= 0, True)
        np.testing.assert_array_equal(keypoints.coordinates[:, 1] <= test_image.height, True)

    def test_scale(self):
        """Tests that the scales are positive."""
        keypoints = self.detector.detect(self.loader.get_image(0))

        if keypoints.scales is not None:
            np.testing.assert_array_equal(keypoints.scales >= 0, True)

    def test_pickleable(self):
        """Tests that the detector object is pickleable (required for dask)."""
        try:
            pickle.dumps(self.detector)
        except TypeError:
            self.fail("Cannot dump detector using pickle")


if __name__ == "__main__":
    unittest.main()
