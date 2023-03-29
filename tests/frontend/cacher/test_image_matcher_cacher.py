"""Unit tests for image matcher catcher.

Authors: John Lambert
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.cacher.image_matcher_cacher import ImageMatcherCacher

DUMMY_KEYPOINTS_I1 = Keypoints(
    coordinates=np.random.rand(10, 2), scales=np.random.rand(10), responses=np.random.rand(10)
)
DUMMY_KEYPOINTS_I2 = Keypoints(
    coordinates=np.random.rand(15, 2), scales=np.random.rand(15), responses=np.random.rand(15)
)

DUMMY_IM_SHAPE_I1 = (100, 200, 3)
DUMMY_IM_SHAPE_I2 = (50, 50, 3)

DUMMY_IMAGE_I1 = Image(value_array=np.zeros(DUMMY_IM_SHAPE_I1, dtype=np.uint8))
DUMMY_IMAGE_I2 = Image(value_array=np.zeros(DUMMY_IM_SHAPE_I2, dtype=np.uint8))

ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent


class TestImageMatcherCacher(unittest.TestCase):
    """Unit tests for ImageMatcherCacher."""

    @patch("gtsfm.utils.cache.generate_hash_for_image", return_value="numpy_key")
    @patch("gtsfm.utils.io.read_from_bz2_file", return_value=None)
    @patch("gtsfm.utils.io.write_to_bz2_file")
    def test_cache_miss(
        self, write_mock: MagicMock, read_mock: MagicMock, generate_hash_for_image_mock: MagicMock
    ) -> None:
        """Test the scenario of cache miss."""

        # mock the underlying detector-descriptor which is used on cache miss
        underlying_matcher_mock = MagicMock()
        underlying_matcher_mock.match.return_value = (DUMMY_KEYPOINTS_I1, DUMMY_KEYPOINTS_I2)
        underlying_matcher_mock.__class__.__name__ = "mock_matcher"
        obj_under_test = ImageMatcherCacher(matcher_obj=underlying_matcher_mock)

        computed_keypoints_i1, computed_keypoints_i2 = obj_under_test.apply(
            image_i1=DUMMY_IMAGE_I1,
            image_i2=DUMMY_IMAGE_I2,
        )
        # assert the returned value
        self.assertEqual(computed_keypoints_i1, DUMMY_KEYPOINTS_I1)
        self.assertEqual(computed_keypoints_i2, DUMMY_KEYPOINTS_I2)

        # assert that underlying object was called
        underlying_matcher_mock.match.assert_called_once_with(
            image_i1=DUMMY_IMAGE_I1,
            image_i2=DUMMY_IMAGE_I2,
        )

        # assert that hash generation was called twice
        generate_hash_for_image_mock.assert_called()

        # assert that read function was called once and write function was called once
        cache_path = ROOT_PATH / "cache" / "image_matcher" / "mock_matcher_numpy_key_numpy_key.pbz2"
        read_mock.assert_called_once_with(cache_path)
        write_mock.assert_called_once_with(
            {"keypoints_i1": DUMMY_KEYPOINTS_I1, "keypoints_i2": DUMMY_KEYPOINTS_I2}, cache_path
        )

    @patch("gtsfm.utils.cache.generate_hash_for_image", return_value="numpy_key")
    @patch(
        "gtsfm.utils.io.read_from_bz2_file",
        return_value={"keypoints_i1": DUMMY_KEYPOINTS_I1, "keypoints_i2": DUMMY_KEYPOINTS_I2},
    )
    @patch("gtsfm.utils.io.write_to_bz2_file")
    def test_cache_hit(
        self, write_mock: MagicMock, read_mock: MagicMock, generate_hash_for_image_mock: MagicMock
    ) -> None:
        """Test the scenario of cache miss."""

        # mock the underlying image matcher which is used on cache miss
        underlying_matcher_mock = MagicMock()
        underlying_matcher_mock.__class__.__name__ = "mock_matcher"
        obj_under_test = ImageMatcherCacher(matcher_obj=underlying_matcher_mock)

        computed_keypoints_i1, computed_keypoints_i2 = obj_under_test.apply(
            image_i1=DUMMY_IMAGE_I1,
            image_i2=DUMMY_IMAGE_I2,
        )
        # assert the returned value
        self.assertEqual(computed_keypoints_i1, DUMMY_KEYPOINTS_I1)
        self.assertEqual(computed_keypoints_i2, DUMMY_KEYPOINTS_I2)

        # assert that underlying object was not called
        underlying_matcher_mock.match.assert_not_called()

        # assert that hash generation was called twice
        generate_hash_for_image_mock.assert_called()

        # assert that read function was called once and write function was called once
        cache_path = ROOT_PATH / "cache" / "image_matcher" / "mock_matcher_numpy_key_numpy_key.pbz2"
        read_mock.assert_called_once_with(cache_path)
        write_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
