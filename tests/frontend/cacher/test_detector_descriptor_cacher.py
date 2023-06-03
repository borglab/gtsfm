"""Unit tests for detector-descriptor cacher.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.cacher.detector_descriptor_cacher import DetectorDescriptorCacher

DUMMY_IMAGE = Image(value_array=np.random.randint(low=0, high=255, size=(100, 120, 3)))

DUMMY_KEYPOINTS = Keypoints(coordinates=np.random.rand(10, 2), scales=np.random.rand(10), responses=np.random.rand(10))
DUMMY_DESCRIPTORS = np.random.rand(len(DUMMY_KEYPOINTS), 128)

ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent


class TestDetectorDescriptorCacher(unittest.TestCase):
    """Unit tests for DetectorDescriptorCacher."""

    @patch("gtsfm.utils.cache.generate_hash_for_image", return_value="img_key")
    @patch("gtsfm.utils.io.read_from_bz2_file", return_value=None)
    @patch("gtsfm.utils.io.write_to_bz2_file")
    def test_cache_miss(
        self, write_mock: MagicMock, read_mock: MagicMock, generate_hash_for_image_mock: MagicMock
    ) -> None:
        """Test the scenario of cache miss."""

        # mock the underlying detector-descriptor which is used on cache miss
        underlying_detector_descriptor_mock = MagicMock()
        underlying_detector_descriptor_mock.detect_and_describe.return_value = (DUMMY_KEYPOINTS, DUMMY_DESCRIPTORS)
        underlying_detector_descriptor_mock.__class__.__name__ = "mock_det_desc"
        obj_under_test = DetectorDescriptorCacher(detector_descriptor_obj=underlying_detector_descriptor_mock)

        computed_keypoints, computed_descriptors = obj_under_test.detect_and_describe(image=DUMMY_IMAGE)
        # assert the returned value
        self.assertEqual(computed_keypoints, DUMMY_KEYPOINTS)
        np.testing.assert_allclose(computed_descriptors, DUMMY_DESCRIPTORS)

        # assert that underlying object was called
        underlying_detector_descriptor_mock.detect_and_describe.assert_called_once_with(DUMMY_IMAGE)

        # assert that hash generation was called with the input image
        generate_hash_for_image_mock.assert_called_with(DUMMY_IMAGE)

        # assert that read function was called once and write function was called once
        cache_path = ROOT_PATH / "cache" / "detector_descriptor" / "mock_det_desc_img_key.pbz2"
        read_mock.assert_called_once_with(cache_path)
        write_mock.assert_called_once_with({"keypoints": DUMMY_KEYPOINTS, "descriptors": DUMMY_DESCRIPTORS}, cache_path)

    @patch("gtsfm.utils.cache.generate_hash_for_image", return_value="img_key")
    @patch(
        "gtsfm.utils.io.read_from_bz2_file",
        return_value={"keypoints": DUMMY_KEYPOINTS, "descriptors": DUMMY_DESCRIPTORS},
    )
    @patch("gtsfm.utils.io.write_to_bz2_file")
    def test_cache_hit(self, write_mock: MagicMock, read_mock: MagicMock, generate_hash_for_image_mock: MagicMock):
        """Test the scenario of cache miss."""

        # mock the underlying detector-descriptor which is used on cache miss
        underlying_detector_descriptor_mock = MagicMock()
        underlying_detector_descriptor_mock.__class__.__name__ = "mock_det_desc"
        obj_under_test = DetectorDescriptorCacher(detector_descriptor_obj=underlying_detector_descriptor_mock)

        computed_keypoints, computed_descriptors = obj_under_test.detect_and_describe(image=DUMMY_IMAGE)
        # assert the returned value
        self.assertEqual(computed_keypoints, DUMMY_KEYPOINTS)
        np.testing.assert_allclose(computed_descriptors, DUMMY_DESCRIPTORS)

        # assert that underlying object was not called
        underlying_detector_descriptor_mock.detect_and_describe.assert_not_called()

        # assert that hash generation was called with the input image
        generate_hash_for_image_mock.assert_called_with(DUMMY_IMAGE)

        # assert that read function was called once and write function was called once
        cache_path = ROOT_PATH / "cache" / "detector_descriptor" / "mock_det_desc_img_key.pbz2"
        read_mock.assert_called_once_with(cache_path)
        write_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
