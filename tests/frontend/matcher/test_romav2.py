"""Unit tests for the RoMa v2 image matcher.

These tests skip when the optional ``romav2`` extra is not installed.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pytest

from gtsfm.loader.olsson_loader import OlssonLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent / "data"
TEST_DATA_PATH = DATA_ROOT_PATH / "set1_lund_door"


def _romav2_available() -> bool:
    try:
        import romav2  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _romav2_available(), reason="optional romav2 package is not installed")
class TestRoMaV2Matcher(unittest.TestCase):
    """Smoke tests for RoMaV2Matcher against the Olsson door pair."""

    @classmethod
    def setUpClass(cls) -> None:
        from gtsfm.frontend.matcher.romav2 import RoMaV2Matcher

        # Use the lightest preset so CI/dev boxes can finish quickly once weights are cached.
        cls.matcher = RoMaV2Matcher(setting="turbo", max_keypoints=512, min_confidence=0.05)
        cls.loader = OlssonLoader(TEST_DATA_PATH, max_resolution=128)

    def test_number_of_keypoints_match(self) -> None:
        image_i0 = self.loader.get_image(0)
        image_i1 = self.loader.get_image(1)

        keypoints_i0, keypoints_i1 = self.matcher.match(image_i0, image_i1)

        self.assertEqual(len(keypoints_i0), len(keypoints_i1))
        self.assertGreater(len(keypoints_i0), 0)


class TestRoMaV2MatcherImport(unittest.TestCase):
    def test_missing_dependency_message(self) -> None:
        """When romav2 is absent, construction should raise a helpful ImportError."""
        if _romav2_available():
            self.skipTest("romav2 is installed in this environment")

        from gtsfm.frontend.matcher.romav2 import RoMaV2Matcher

        with self.assertRaises(ImportError) as context:
            RoMaV2Matcher()
        self.assertIn("romav2", str(context.exception))


if __name__ == "__main__":
    unittest.main()
