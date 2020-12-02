"""Tests for Keypoints class.

Authors: Ayush Baid
"""
import random
import unittest

import numpy as np

from common.keypoints import Keypoints

# Creating dummy data for tests
NUM_ENTRIES = random.randint(1, 10)
COORDINATES = np.random.rand(NUM_ENTRIES, 2)
SCALES = np.random.rand(NUM_ENTRIES)
RESPONSES = np.random.rand(NUM_ENTRIES)


class TestKeypoints(unittest.TestCase):
    """Unit tests for Keypoints class."""

    def test_constructor_with_coordinates_only(self):
        """Tests the construction of keypoints with just coordinates."""

        result = Keypoints(coordinates=COORDINATES)

        np.testing.assert_array_equal(result.coordinates, COORDINATES)
        self.assertIsNone(result.responses)
        self.assertIsNone(result.scales)

    def test_constructor_with_all_inputs(self):
        """Tests the construction of keypoints with all data."""

        result = Keypoints(coordinates=COORDINATES,
                           scales=SCALES,
                           responses=RESPONSES)

        np.testing.assert_array_equal(result.coordinates, COORDINATES)
        np.testing.assert_array_equal(result.scales, SCALES)
        np.testing.assert_array_equal(result.responses, RESPONSES)

    def test_equality(self):
        """Tests the equality checker."""

        # Test with None scales and responses.
        obj1 = Keypoints(coordinates=COORDINATES)
        obj2 = Keypoints(coordinates=COORDINATES)
        self.assertEqual(obj1, obj2)

        # test with coordinates and scales.
        obj1 = Keypoints(coordinates=COORDINATES,
                         scales=SCALES,
                         responses=RESPONSES)
        obj2 = Keypoints(coordinates=COORDINATES,
                         scales=SCALES,
                         responses=RESPONSES)
        self.assertEqual(obj1, obj2)

        # Test with one object having scales and other not having scales.
        obj1 = Keypoints(coordinates=COORDINATES, scales=SCALES)
        obj2 = Keypoints(coordinates=COORDINATES)
        print(obj1 != obj2)
        self.assertNotEqual(obj1, obj2)

        # Test with one object having responses and other not having responses.
        obj1 = Keypoints(coordinates=COORDINATES, responses=RESPONSES)
        obj2 = Keypoints(coordinates=COORDINATES)
        self.assertNotEqual(obj1, obj2)


if __name__ == "__main__":
    unittest.main()
