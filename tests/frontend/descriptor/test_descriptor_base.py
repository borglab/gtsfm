"""Tests for frontend's base descriptor class.

Authors: Ayush Baid
"""
import pickle
import unittest
from pathlib import Path

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
        self.loader = OlssonLoader(str(TEST_DATA_PATH))

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

    def test_pickleable(self):
        """Tests that the descriptor is pickleable (required for dask)."""
        try:
            pickle.dumps(self.descriptor)
        except TypeError:
            self.fail("Cannot dump descriptor using pickle")


if __name__ == "__main__":
    unittest.main()
