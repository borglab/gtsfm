"""
Tests for frontend's base detector class.

Authors: Ayush Baid
"""

import unittest

import dask
import numpy as np

from frontend.detector.dummy_detector import DummyDetector
from loader.folder_loader import FolderLoader

# defining the path for test data
TEST_DATA_PATH = 'tests/data/lund'


class TestDetectorBase(unittest.TestCase):
    """Main test class for detector base class in frontend"""

    def setUp(self):
        super().setUp()
        self.detector = DummyDetector()
        self.loader = FolderLoader(TEST_DATA_PATH)

    def test_coordinates(self):
        """
        Tests that each coordinate is within the image bounds
        """
        test_image = self.loader.get_image(0)
        features = self.detector.detect(test_image)

        image_shape = test_image.image_array.shape

        np.testing.assert_array_equal(features[:, 0] >= 0, True)
        np.testing.assert_array_equal(features[:, 0] <= image_shape[1], True)
        np.testing.assert_array_equal(features[:, 1] >= 0, True)
        np.testing.assert_array_equal(features[:, 1] <= image_shape[0], True)

    def test_scale(self):
        """
        Tests that the scales are positive
        """
        features = self.detector.detect(self.loader.get_image(0))

        np.testing.assert_array_equal(features[:, 2] >= 0, True)

    def test_num_columns(self):
        """
        Tests the number of columns in the features are >=2
        """
        features = self.detector.detect(self.loader.get_image(0))

        if features.size > 0:
            self.assertLessEqual(2, features.shape[1])

    def test_computation_graph(self):
        """
        Test the dask's computation graph formation using a single image
        """

        loader_graph = self.loader.create_computation_graph()
        detector_graph = self.detector.create_computation_graph(loader_graph)
        results = dask.compute(detector_graph)[0]

        # check the number of results
        self.assertEqual(len(results), len(self.loader),
                         "Dask workflow does not return the same number of results"
                         )

        # check the results via normal workflow and dask workflow for an image
        normal_features = self.detector.detect(self.loader.get_image(0))
        dask_features = results[0]
        np.testing.assert_allclose(normal_features, dask_features)
