"""Unit tests for cache utils.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path

import numpy as np

import gtsfm.utils.cache as cache_utils
import gtsfm.utils.io as io_utils

TEST_DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"


class TestCacheUtils(unittest.TestCase):
    """Unit tests for cache utils."""

    def test_generate_hash_for_numpy_array(self):
        numpy_arr = np.array([1.0, 2.0, 3.5, -1.0, -2.0, 1.0]).reshape((2, 3))
        key = cache_utils.generate_hash_for_numpy_array(numpy_arr)
        expected = "22ed44b44ded282578e67f7d4c759038be79db5b"
        self.assertEqual(key, expected)

    def test_generate_hash_for_image(self):
        image_path = TEST_DATA_ROOT_PATH / "set1_lund_door" / "images" / "DSC_0001.JPG"
        image = io_utils.load_image(str(image_path))
        key = cache_utils.generate_hash_for_image(image=image)
        expected = "033223b24a9edfe6e989b6853db295df51011f3ce3b4545791778669f08d15213223f7873ea8dcdd"
        self.assertEqual(key, expected)


if __name__ == "__main__":
    unittest.main()
