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

    def test_cast_to_opencv_keypoints(self):
        """Tests conversion of GTSFM's keypoints to OpenCV's keypoints."""

        gtsfm_keypoints = Keypoints(
            coordinates=np.array([
                [1.3, 5],
                [20, 10]
            ]),
            scales=np.array([1.0, 5.2]),
            responses=np.array([4.2, 3.2]))

        results = gtsfm_keypoints.cast_to_opencv_keypoints()

        # check the length of the result
        self.assertEqual(len(results), len(gtsfm_keypoints))

        # check all the keypoint values
        for idx in range(len(gtsfm_keypoints)):

            opencv_kp = results[idx]
            self.assertAlmostEqual(
                opencv_kp.pt[0], gtsfm_keypoints.coordinates[idx, 0])
            self.assertAlmostEqual(
                opencv_kp.pt[1], gtsfm_keypoints.coordinates[idx, 1])
            self.assertAlmostEqual(
                opencv_kp.size, gtsfm_keypoints.scales[idx])
            self.assertAlmostEqual(
                opencv_kp.response, gtsfm_keypoints.responses[idx])


if __name__ == "__main__":
    unittest.main()
