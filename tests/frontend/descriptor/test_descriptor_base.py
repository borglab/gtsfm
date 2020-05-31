"""
Tests for frontend's base descriptor class.

Authors: Ayush Baid
"""
import unittest

import dask
import numpy as np

from common.image import Image
from frontend.descriptor.descriptor_base import DescriptorBase
from loader.folder_loader import FolderLoader

# defining the path for test data
TEST_DATA_PATH = 'tests/data/lund'


class DummyDescriptor(DescriptorBase):
    """
    A dummy descriptor which assigns random descriptor
    """

    def __init__(self):
        super().__init__()

        self.descriptor_length = 15  # length of each descriptor

    def describe(self, image: Image, features: np.ndarray) -> np.ndarray:
        """
        Assign descriptors at each feature

        Arguments:
            image (Image): the input image
            features (np.ndarray): the features to describe

        Returns:
            np.ndarray: the descriptors for the input features
        """
        np.random.seed(0)

        if features.size == 0:
            return np.array([])

        return np.random.rand(features.shape[0], self.descriptor_length)


class TestDescriptorBase(unittest.TestCase):
    """
    Unit tests for the Base descriptor class.
    Should be inherited by all descriptor unit tests.
    """

    def setUp(self):
        self.descriptor = DummyDescriptor()
        self.loader = FolderLoader(TEST_DATA_PATH)

    def test_result_size(self):
        """
        Check if the number of descriptors are same as number of features
        """
        input_image = self.loader.get_image(0)
        input_features = np.random.randint(
            low=[0, 0], high=input_image.get_shape(), size=(5, 2)
        )

        result = self.descriptor.describe(input_image, input_features)

        self.assertEqual(input_features.shape[0], result.shape[0])

    def test_no_features(self):
        """
        Checks that empty feature inputs works well
        """
        input_image = self.loader.get_image(0)
        input_features = np.array([])

        result = self.descriptor.describe(input_image, input_features)

        self.assertEqual(0, result.size)

    def test_create_computation_graph(self):
        """
        Checks the dask computation graph
        """

        # testing some indices
        test_indices = [0, 5]
        test_images = [self.loader.get_image(idx) for idx in test_indices]
        test_features = [np.random.randint(
            low=[0, 0], high=x.get_shape(), size=(np.random.randint(5, 10), 2)
        ) for x in test_images]

        description_graph = self.descriptor.create_computation_graph(
            [dask.delayed(x) for x in test_images],
            [dask.delayed(x) for x in test_features]
        )

        description_results = dask.compute(description_graph)[0]

        for idx in range(len(test_indices)):
            np.testing.assert_allclose(
                self.descriptor.describe(test_images[idx], test_features[idx]),
                description_results[idx]
            )
