"""Unit tests for MASt3R correspondence generator.

Authors: Akshay Krishnan
"""

from pathlib import Path
import unittest

from dask.distributed import Client, LocalCluster
from PIL import Image

from gtsfm.frontend.correspondence_generator.mast3r_correspondence_generator import Mast3rCorrespondenceGenerator
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.utils import viz as viz_utils

# defining the path for test data
DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent / "data"
TEST_DATA_PATH = DATA_ROOT_PATH / "set1_lund_door"

IMAGE_PAIRS = [(0, 1), (1, 2)]

SAVE_VISUALIZATION = False  # Use for debugging locally.


class TestMast3rCorrespondenceGenerator(unittest.TestCase):

    def setUp(self) -> None:
        cluster = LocalCluster(n_workers=1, threads_per_worker=1, memory_limit="32GB")
        self._client = Client(cluster)

        self._loader = OlssonLoader(str(TEST_DATA_PATH), max_frame_lookahead=4)

    def test_data(self) -> None:
        corr_gen = Mast3rCorrespondenceGenerator()

        keypoints, match_indices = corr_gen.generate_correspondences(
            self._client, self._loader.get_all_images_as_futures(self._client), IMAGE_PAIRS
        )

        self.assertEqual(len(keypoints), 3)  # Corresponds to images 0, 1, 2 used in IMAGE_PAIRS
        self.assertEqual(len(match_indices), len(IMAGE_PAIRS))

        # Assert that we get a min of 100 keypoints and matches, as it was an easy dataset
        for i, kpts in enumerate(keypoints):
            # Only check for keypoints on images that were actually processed (0, 1, 2)
            if i in [0, 1, 2]:
                self.assertGreaterEqual(len(kpts), 100, f"Image {i} has {len(kpts)} < 100 of keypoints")

        for pair_tuple, corr_idxs in match_indices.items():
            self.assertGreaterEqual(len(corr_idxs), 100, f"Pair {pair_tuple} has {len(corr_idxs)} < 100 matches")

        if SAVE_VISUALIZATION:
            img1 = self._loader.get_image(0)
            img2 = self._loader.get_image(1)

            viz_image = viz_utils.plot_twoview_correspondences(
                img1, img2, keypoints[0], keypoints[1], match_indices[(0, 1)]
            )
            Image.fromarray(viz_image.value_array).save("mast3r_testplot.png")


if __name__ == "__main__":
    unittest.main()
