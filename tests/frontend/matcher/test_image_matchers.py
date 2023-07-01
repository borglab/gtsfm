"""Unit tests for image matchers

Authors: Ayush Baid
"""
import unittest
from pathlib import Path

import parameterized

from gtsfm.frontend.matcher.loftr import LOFTR
from gtsfm.loader.olsson_loader import OlssonLoader

# defining the path for test data
DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent / "data"
TEST_DATA_PATH = DATA_ROOT_PATH / "set1_lund_door"


@parameterized.parameterized_class(
    [{"matcher": LOFTR(use_outdoor_model=True)}, {"matcher": LOFTR(use_outdoor_model=False)}]
)
class TestImageMatchers(unittest.TestCase):
    matcher = LOFTR(use_outdoor_model=True)

    def setUp(self) -> None:
        super().setUp()
        self.loader = OlssonLoader(TEST_DATA_PATH, image_extension="JPG", max_resolution=128)

    def test_number_of_keypoints_match(self):
        image_i0 = self.loader.get_image(0)
        image_i1 = self.loader.get_image(1)

        keypoints_i0, keypoints_i1 = self.matcher.match(image_i0, image_i1)

        self.assertEqual(len(keypoints_i0), len(keypoints_i1))
