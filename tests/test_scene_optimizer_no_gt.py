"""Unit tests for the scene-optimizer class.

Authors: Ayush Baid, John Lambert
"""
import unittest
from pathlib import Path

from tests.test_scene_optimizer import TestSceneOptimizer
from gtsfm.loader.olsson_loader import OlssonLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent / "data"


class TestSceneOptimizerNoGT(TestSceneOptimizer):
    """Unit test for SceneOptimizer without any ground truth data available, which runs SfM for a scene."""

    def setUp(self) -> None:
        self.loader = OlssonLoader(
            str(DATA_ROOT_PATH / "set3_lund_door_nointrinsics_noextrinsics"), image_extension="JPG"
        )
        assert len(self.loader)


if __name__ == "__main__":
    unittest.main()
