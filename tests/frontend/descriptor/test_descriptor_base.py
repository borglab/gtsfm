"""Tests for frontend's base descriptor class.

Authors: Ayush Baid
"""
import pickle
import unittest
from pathlib import Path

import dask
import numpy as np

from common.keypoints import Keypoints
from frontend.descriptor.dummy_descriptor import DummyDescriptor
from loader.folder_loader import FolderLoader

# defining the path for test data
DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent / 'data'
TEST_DATA_PATH = DATA_ROOT_PATH / 'set1'


class TestDescriptorBase(unittest.TestCase):
    """Unit tests for the DescriptorBase class.

    Should be inherited by all descriptor unit tests.
    """

    def setUp(self):
        self.descriptor = DummyDescriptor()
        self.loader = FolderLoader(str(TEST_DATA_PATH), image_extension='JPG')

    def test_result_size(self):
        """Check if the number of descriptors are same as number of features."""

        input_image = self.loader.get_image(0)
        input_keypoints = Keypoints(coordinates=np.random.randint(
            low=[0, 0],
            high=[input_image.width, input_image.height],
            size=(5, 2)
        ))

        result = self.descriptor.describe(input_image, input_keypoints)

        self.assertEqual(len(input_keypoints), result.shape[0])

    def test_with_no_features(self):
        """Checks that empty feature inputs works well."""
        input_image = self.loader.get_image(0)
        input_keypoints = Keypoints(coordinates=np.array([]))

        result = self.descriptor.describe(input_image, input_keypoints)

        self.assertEqual(0, result.size)

    def test_create_computation_graph(self):
        """Checks the dask computation graph."""

        # testing some indices
        test_indices = [0, 5]
        test_images = [self.loader.get_image(idx) for idx in test_indices]
        test_keypoints = [Keypoints(coordinates=np.random.randint(
            low=[0, 0],
            high=[x.width, x.height],
            size=(np.random.randint(5, 10), 2)
        )) for x in test_images]

        description_graph = self.descriptor.create_computation_graph(
            [dask.delayed(x) for x in test_images],
            [dask.delayed(x) for x in test_keypoints]
        )

        with dask.config.set(scheduler='single-threaded'):
            description_results = dask.compute(description_graph)[0]

        for idx in range(len(test_indices)):
            np.testing.assert_allclose(
                self.descriptor.describe(test_images[idx], test_keypoints[idx]),
                description_results[idx]
            )

    def test_pickleable(self):
        """Tests that the descriptor is pickleable (required for dask)."""
        try:
            pickle.dumps(self.descriptor)
        except TypeError:
            self.fail("Cannot dump descriptor using pickle")


if __name__ == '__main__':
    unittest.main()
