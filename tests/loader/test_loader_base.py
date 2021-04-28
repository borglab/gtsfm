"""Tests for defined methods for LoaderBase.

Authors: Ayush Baid
"""
import unittest
from typing import Optional

from gtsam import Cal3Bundler, Pose3

from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase


class DummyLoader(LoaderBase):
    def __init__(self) -> None:
        super().__init__()

        self._valid_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 2)]

    def __len__(self) -> int:
        return 4

    def validate_pair(self, idx1: int, idx2: int) -> bool:
        return (idx1, idx2) in self._valid_pairs

    def get_camera_intrinsics(self, index: int) -> Optional[Cal3Bundler]:
        raise NotImplementedError

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        raise NotImplementedError

    def get_image(self, index: int) -> Image:
        raise NotImplementedError


class TestLoaderBase(unittest.TestCase):
    """Unit tests for methods in LoaderBase."""

    def test_get_valid_pairs(self):
        loader = DummyLoader()

        computed = loader.get_valid_pairs()
        expected = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 2)]

        self.assertListEqual(computed, expected)

    def test_get_valid_triplets(self):
        loader = DummyLoader()

        computed = loader.get_valid_triplets()
        expected = [
            (0, 1, 2),
            (1, 2, 3),
        ]

        self.assertListEqual(computed, expected)


if __name__ == "__main__":
    unittest.main()
