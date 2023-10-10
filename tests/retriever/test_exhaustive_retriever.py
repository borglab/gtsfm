"""Unit tests for the exhaustive-matching retriever.

Authors: John Lambert
"""

import unittest
from pathlib import Path

from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.retriever.exhaustive_retriever import ExhaustiveRetriever

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
DOOR_DATA_ROOT = DATA_ROOT_PATH / "set1_lund_door"


class TestExhaustiveRetriever(unittest.TestCase):
    def test_exhaustive_retriever_door(self) -> None:
        """Test the Exhaustive retriever on 12 frames of the Lund Door Dataset."""
        loader = OlssonLoader(folder=str(DOOR_DATA_ROOT))
        retriever = ExhaustiveRetriever()

        pairs = retriever.get_image_pairs(
            global_descriptors=None, image_fnames=loader.image_filenames(), plots_output_dir=None
        )

        # {12 \choose 2} = (12 * 11) / 2 = 66
        self.assertEqual(len(pairs), 66)

        for i1, i2 in pairs:
            self.assertTrue(i1 < i2)


if __name__ == "__main__":
    unittest.main()
