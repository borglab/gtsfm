"""Tests for Keypoints class.

Authors: Ayush Baid
"""
import random
import unittest

import numpy as np

from gtsfm.common.keypoints import Keypoints

# Creating dummy data for tests
NUM_ENTRIES = random.randint(1, 10)
COORDINATES = np.random.rand(NUM_ENTRIES, 2)
SCALES = np.random.rand(NUM_ENTRIES)
RESPONSES = np.random.rand(NUM_ENTRIES)


class TestKeypoints(unittest.TestCase):
    """Unit tests for Keypoints class."""

    def compare_without_ordering(self, keypoints1: Keypoints, keypoints2: Keypoints) -> bool:
        # compare that values of the keypoints in an order insensitive way
        list_of_tuples1 = []
        for idx in range(len(keypoints1)):
            list_of_tuples1.append(
                (
                    keypoints1.coordinates[idx, 0],
                    keypoints1.coordinates[idx, 1],
                    None if keypoints1.scales is None else keypoints1.scales[idx],
                    None if keypoints1.responses is None else keypoints1.responses[idx],
                )
            )

        list_of_tuples2 = []
        for idx in range(len(keypoints2)):
            list_of_tuples2.append(
                (
                    keypoints2.coordinates[idx, 0],
                    keypoints2.coordinates[idx, 1],
                    None if keypoints2.scales is None else keypoints2.scales[idx],
                    None if keypoints2.responses is None else keypoints2.responses[idx],
                )
            )

        self.assertCountEqual(list_of_tuples1, list_of_tuples2)

    def test_constructor_with_coordinates_only(self):
        """Tests the construction of keypoints with just coordinates."""

        result = Keypoints(coordinates=COORDINATES)

        np.testing.assert_array_equal(result.coordinates, COORDINATES)
        self.assertIsNone(result.responses)
        self.assertIsNone(result.scales)

    def test_constructor_with_all_inputs(self):
        """Tests the construction of keypoints with all data."""

        result = Keypoints(coordinates=COORDINATES, scales=SCALES, responses=RESPONSES)

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
        obj1 = Keypoints(coordinates=COORDINATES, scales=SCALES, responses=RESPONSES)
        obj2 = Keypoints(coordinates=COORDINATES, scales=SCALES, responses=RESPONSES)
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

    def test_get_top_k_with_responses(self):
        """Tests the selection of top entries in a keypoints with responses."""

        input_keypoints = Keypoints(
            coordinates=np.array(
                [
                    [10.0, 23.2],
                    [37.1, 50.2],
                    [90.1, 10.7],
                    [150.0, 122.0],
                    [250.0, 49.0],
                ]
            ),
            scales=np.array([1, 3, 2, 3.2, 1.8]),
            responses=np.array([0.3, 0.7, 0.9, 0.1, 0.2]),
        )

        # test with requested length > current length
        requested_length = len(input_keypoints) * 2
        computed, _ = input_keypoints.get_top_k(requested_length)

        self.assertEqual(computed, input_keypoints)

        # test with requested length < current length
        requested_length = 2
        computed, _ = input_keypoints.get_top_k(requested_length)

        expected = Keypoints(
            coordinates=np.array(
                [
                    [37.1, 50.2],
                    [90.1, 10.7],
                ]
            ),
            scales=np.array([3, 2]),
            responses=np.array([0.7, 0.9]),
        )

        # compare in an order-insensitive fashion
        self.compare_without_ordering(computed, expected)

    def test_get_top_k_without_responses(self):
        """Tests the selection of top entries in a keypoints w/o responses."""

        input_keypoints = Keypoints(
            coordinates=np.array(
                [
                    [10.0, 23.2],
                    [37.1, 50.2],
                    [90.1, 10.7],
                    [150.0, 122.0],
                    [250.0, 49.0],
                ]
            )
        )

        # test with requested length > current length
        requested_length = len(input_keypoints) * 2
        computed, _ = input_keypoints.get_top_k(requested_length)

        self.assertEqual(computed, input_keypoints)

        # test with requested length < current length
        requested_length = 2
        computed, _ = input_keypoints.get_top_k(requested_length)

        expected = Keypoints(coordinates=input_keypoints.coordinates[:requested_length])

        self.assertEqual(computed, expected)

    def test_filter_by_mask(self) -> None:
        """Test the `filter_by_mask` method."""
        # Create a (9, 9) mask with ones in a (5, 5) square in the center of the mask and zeros everywhere else.
        mask = np.zeros((9, 9)).astype(np.uint8)
        mask[2:7, 2:7] = 1

        # Test coordinates near corners of square of ones and along the diagonal.
        coordinates = np.array(
            [
                [1.4, 1.4],
                [1.4, 6.4],
                [6.4, 1.4],
                [6.4, 6.4],
                [5.0, 5.0],
                [0.0, 0.0],
                [8.0, 8.0],
            ]
        )
        input_keypoints = Keypoints(coordinates=coordinates)
        expected_keypoints = Keypoints(coordinates=coordinates[[3, 4]])

        # Create keypoints from coordinates and dummy descriptors.
        filtered_keypoints, _ = input_keypoints.filter_by_mask(mask)
        assert len(filtered_keypoints) == 2
        self.assertEqual(filtered_keypoints, expected_keypoints)

    def test_cast_to_opencv_keypoints(self):
        """Tests conversion of GTSFM's keypoints to OpenCV's keypoints."""

        gtsfm_keypoints = Keypoints(
            coordinates=np.array([[1.3, 5], [20, 10]]),
            scales=np.array([1.0, 5.2]),
            responses=np.array([4.2, 3.2]),
        )

        results = gtsfm_keypoints.cast_to_opencv_keypoints()

        # check the length of the result
        self.assertEqual(len(results), len(gtsfm_keypoints))

        # check all the keypoint values
        for idx in range(len(gtsfm_keypoints)):

            opencv_kp = results[idx]
            self.assertAlmostEqual(opencv_kp.pt[0], gtsfm_keypoints.coordinates[idx, 0], places=5)
            self.assertAlmostEqual(opencv_kp.pt[1], gtsfm_keypoints.coordinates[idx, 1], places=5)
            self.assertAlmostEqual(opencv_kp.size, gtsfm_keypoints.scales[idx], places=5)
            self.assertAlmostEqual(opencv_kp.response, gtsfm_keypoints.responses[idx], places=5)

    def test_extract_indices_valid(self):
        """Test extraction of indices."""

        # test without scales and responses
        input = Keypoints(coordinates=np.array([[1.3, 5], [20, 10], [5.0, 1.3], [2.1, 4.2]]))
        indices = np.array([0, 2])

        expected = Keypoints(coordinates=np.array([[1.3, 5], [5.0, 1.3]]))
        computed = input.extract_indices(indices)

        self.assertEqual(computed, expected)

        # test without scales and responses
        input = Keypoints(
            coordinates=np.array([[1.3, 5], [20, 10], [5.0, 1.3], [2.1, 4.2]]),
            scales=np.array([0.2, 0.5, 0.3, 0.9]),
            responses=np.array([2.3, 1.2, 4.5, 0.2]),
        )
        indices = np.array([0, 2])

        expected = Keypoints(
            coordinates=np.array([[1.3, 5], [5.0, 1.3]]), scales=np.array([0.2, 0.3]), responses=np.array([2.3, 4.5])
        )
        computed = input.extract_indices(indices)

        self.assertEqual(computed, expected)

    def test_extract_indices_empty(self):
        """Test extraction of indices, which are empty."""

        # test without scales and responses
        input = Keypoints(coordinates=np.array([[1.3, 5], [20, 10], [5.0, 1.3], [2.1, 4.2]]))
        indices = np.array([])

        computed = input.extract_indices(indices)

        self.assertEqual(len(computed), 0)


if __name__ == "__main__":
    unittest.main()
