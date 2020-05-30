"""
Tests for frontend's base detector class.

Authors: Ayush Baid
"""

import unittest

import dask
import numpy as np

from frontend.detector.detector_base import DetectorBase
from loader.folder_loader import FolderLoader

# defining the path for test data
TEST_DATA_PATH = 'tests/data/lund'


class DummyDetector(DetectorBase):
    """
    A dummy detector which returns random features
    """

    def detect(self, image):
        # randomly decide the number of features
        num_features = 6

        """
        Fill in the columns with random coordinates, scale and optional extra columns

        Constraints:
        1. Coordinates must within the image
        2. scale must be non-negative
        """
        np.random.seed(0)

        num_columns = 4

        features = np.empty((num_features, num_columns))

        # assign the coordinates
        features[:, :2] = np.random.randint(
            [0, 0], high=[image.image_array.shape[1], image.image_array.shape[0]], size=(num_features, 2))

        # assign the scale
        features[:, 2] = np.random.rand(num_features)

        # assign other dimensions independently
        if num_columns > 3:
            features[:, 3:] = np.random.rand(num_features, num_columns-3)

        return features


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

    def test_computation_graph(self):
        """
        Test the dask's computation graph formation using a single image
        """

        loader_graph = self.loader.create_computation_graph()
        detector_graph = self.detector.create_computation_graph(loader_graph)
        results = dask.compute(detector_graph)[0]
        # import pdb
        # pdb.set_trace()

        # check the number of results
        self.assertEqual(len(results), len(self.loader),
                         "Dask workflow does not return the same number of results"
                         )

        # check the
        normal_features = self.detector.detect(self.loader.get_image(0))
        dask_features = results[0]
        np.testing.assert_allclose(normal_features, dask_features)
