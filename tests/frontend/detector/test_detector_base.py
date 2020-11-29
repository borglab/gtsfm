"""
Tests for frontend's base detector class.

Authors: Ayush Baid
"""
import pickle
import unittest

import dask
import numpy as np

from frontend.detector.dummy_detector import DummyDetector
from loader.folder_loader import FolderLoader

# defining the path for test data
TEST_DATA_PATH = 'tests/data/lund'


class TestDetectorBase(unittest.TestCase):
    """Main test class for detector base class in frontend."""

    def setUp(self):
        super().setUp()
        self.detector = DummyDetector()
        self.loader = FolderLoader(TEST_DATA_PATH)

    def test_number_of_detections(self):
        """Tests that the number of detections is less than the maximum number
        configured."""
        test_image = self.loader.get_image(0)
        keypoints = self.detector.detect(test_image)

        self.assertLessEqual(len(keypoints), self.detector.max_keypoints)

    def test_coordinates_range(self):
        """Tests that each coordinate is within the image bounds."""
        test_image = self.loader.get_image(0)
        keypoints = self.detector.detect(test_image)

        np.testing.assert_array_equal(
            keypoints.coordinates[:, 0] >= 0, True)
        np.testing.assert_array_equal(
            keypoints.coordinates[:, 0] <= test_image.width, True)
        np.testing.assert_array_equal(
            keypoints.coordinates[:, 1] >= 0, True)
        np.testing.assert_array_equal(
            keypoints.coordinates[:, 1] <= test_image.height, True)

    def test_scale(self):
        """Tests that the scales are positive."""
        keypoints = self.detector.detect(self.loader.get_image(0))

        np.testing.assert_array_equal(keypoints.scale >= 0, True)

    def test_computation_graph(self):
        """Test the dask's computation graph formation using a single image."""

        image_graph = self.loader.create_computation_graph_for_images()
        detector_graph = self.detector.create_computation_graph(image_graph)

        results = []
        with dask.config.set(scheduler='single-threaded'):
            results = dask.compute(detector_graph)[0]

        # check the number of results
        self.assertEqual(len(results), len(self.loader),
                         "Dask workflow does not return the same number of results"
                         )

        # check the results via normal workflow and dask workflow for an image
        expected_keypoints = self.detector.detect(self.loader.get_image(0))
        computed_keypoints = results[0]

        np.testing.assert_allclose(
            computed_keypoints.coordinates, expected_keypoints.coordinates)
        np.testing.assert_allclose(
            computed_keypoints.scale, expected_keypoints.scale)
        np.testing.assert_allclose(
            computed_keypoints.response, expected_keypoints.response)

    def test_pickleable(self):
        """Tests that the detector object is pickleable (required for dask)."""
        try:
            pickle.dumps(self.detector)
        except TypeError:
            self.fail("Cannot dump detector using pickle")


if __name__ == '__main__':
    unittest.main()
