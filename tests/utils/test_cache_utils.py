"""Unit tests for cache utils.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path

import numpy as np
from gtsfm.common.image import Image

import gtsfm.utils.cache as cache_utils
import gtsfm.utils.io as io_utils

TEST_DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"


class TestCacheUtils(unittest.TestCase):
    """Unit tests for cache utils."""

    def test_generate_hash_for_numpy_array(self):
        numpy_arr = np.random.randn(100, 200)
        key = cache_utils.generate_hash_for_numpy_array(numpy_arr)

    def test_generate_hash_for_image(self):
        image_path = TEST_DATA_ROOT_PATH / "set1_lund_door" / "images" / "DSC_0001.JPG"
        image = io_utils.load_image(str(image_path))

        key = cache_utils.generate_hash_for_image(image=image)


if __name__ == "__main__":
    unittest.main()
