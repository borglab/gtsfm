"""
Tests for frontend's base detector class.

Authors: Ayush Baid
"""

import os
import unittest

import dask
import numpy as np

import utils.io as io
from frontend.detector.detector_base import DetectorBase
from loader.loader_base import LoaderBase


class DummyLoader(LoaderBase):
    """
    Dummy loader with a single lenna image
    """

    def __init__(self):
        super().__init__()

        img_paths = [
            os.path.abspath('tests/data/images/lenna.jpg')
        ]

        self.images = [io.load_image(x) for x in img_paths]

    def __len__(self):
        return 1

    def get_image(self, index: int):
        if index < 0 or index >= self.__len__():
            raise IndexError("Image index is invalid")

        return self.images[index]


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
            [0, 0], high=[image.shape[1], image.shape[0]], size=(num_features, 2))

        # assign the scale
        features[:, 2] = np.random.rand(num_features)

        # assign other dimensions independently
        if num_columns > 3:
            features[:, 3:] = np.random.rand(num_features, num_columns-3)

        return features


class TestDetectorBase(unittest.TestCase):
    """Main test class for detector base class in frontend"""

    def setUp(self):
        self.detector = DummyDetector()

    def test_coordinates(self):
        """
        Tests that each coordinate is within the image bounds
        """
        im_path = os.path.abspath('tests/data/images/lenna.jpg')
        test_image = io.load_image(im_path)
        features = self.detector.detect(test_image)

        assert np.all(np.logical_and(
            features[:, 0] >= 0,
            features[:, 0] <= test_image.shape[1]))
        assert np.all(np.logical_and(
            features[:, 1] >= 0,
            features[:, 1] <= test_image.shape[0]))

    def test_scale(self):
        """
        Tests that the scales are positive
        """
        im_path = os.path.abspath('tests/data/images/lenna.jpg')
        test_image = io.load_image(im_path)
        features = self.detector.detect(test_image)

        assert np.all(features[:, 2] >= 0)

    def test_computation_graph(self):
        """
        Test the dask's computation graph formation using a single image
        """

        loader = DummyLoader()

        loader_graph = loader.create_computation_graph_all()
        detector_graph = self.detector.create_computation_graph(loader_graph)
        results = dask.compute(detector_graph)

        # check the number of results
        assert len(results) == loader.__len__()

        # check the
        assert np.allclose(self.detector.detect(
            loader.get_image(0)), results[0])
