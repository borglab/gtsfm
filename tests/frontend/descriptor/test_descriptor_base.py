"""Tests for frontend's base descriptor class.

Authors: Ayush Baid
"""
import pickle
import unittest
from pathlib import Path

import dask
import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.descriptor.dummy_descriptor import DummyDescriptor
from gtsfm.loader.olsson_loader import OlssonLoader

# defining the path for test data
DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent / "data"
TEST_DATA_PATH = DATA_ROOT_PATH / "set1_lund_door"


class TestDescriptorBase(unittest.TestCase):
    """Unit tests for the DescriptorBase class.

    Should be inherited by all descriptor unit tests.
    """

    def setUp(self):
        self.descriptor = DummyDescriptor()
        self.loader = OlssonLoader(str(TEST_DATA_PATH), image_extension="JPG")

    def test_result_size(self):
        """Check if the number of descriptors are same as number of features."""

        input_image = self.loader.get_image(0)
        input_keypoints = Keypoints(
            coordinates=np.random.randint(
                low=[0, 0],
                high=[input_image.width, input_image.height],
                size=(5, 2),
            )
        )

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
        idxs_under_test = [0, 5]

        for idx in idxs_under_test:

            test_image = self.loader.get_image(idx)
            test_keypoints = Keypoints(
                coordinates=np.random.randint(
                    low=[0, 0],
                    high=[test_image.width, test_image.height],
                    size=(np.random.randint(5, 10), 2),
                )
            )

            descriptor_graph = self.descriptor.create_computation_graph(
                dask.delayed(test_image),
                dask.delayed(test_keypoints),
            )

            with dask.config.set(scheduler="single-threaded"):
                descriptors = dask.compute(descriptor_graph)[0]

            expected_descriptors = self.descriptor.describe(test_image, test_keypoints)

            np.testing.assert_allclose(descriptors, expected_descriptors)

    def test_pickleable(self):
        """Tests that the descriptor is pickleable (required for dask)."""
        try:
            pickle.dumps(self.descriptor)
        except TypeError:
            self.fail("Cannot dump descriptor using pickle")


if __name__ == "__main__":
    unittest.main()
