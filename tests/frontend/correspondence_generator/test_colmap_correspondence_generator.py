"""Unit tests for Colmap correspondence generators.

Authors: Ayush Baid.
"""

from pathlib import Path
import unittest

from dask.distributed import Client

from gtsfm.frontend.correspondence_generator.colmap_correspondence_generator import ColmapCorrespondenceGenerator
from gtsfm.loader.olsson_loader import OlssonLoader

# defining the path for test data
DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent / "data"
TEST_DATA_PATH = DATA_ROOT_PATH / "set1_lund_door"
DB_PATH = TEST_DATA_PATH / "colmap.db"

IMAGE_PAIRS = [(0, 1), (1, 2), (2, 8), (3, 2)]


class TestColmapCorrespondenceGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self._client = Client()

        self._loader = OlssonLoader(str(TEST_DATA_PATH), max_frame_lookahead=4)

    def test_data(self) -> None:
        corr_gen = ColmapCorrespondenceGenerator(str(DB_PATH))

        keypoints, match_indices = corr_gen.generate_correspondences(
            self._client, self._loader.get_all_images_as_futures(self._client), IMAGE_PAIRS
        )

        self.assertEqual(len(keypoints), len(self._loader))
        self.assertEqual(len(match_indices), len(IMAGE_PAIRS))


if __name__ == "__main__":
    unittest.main()
