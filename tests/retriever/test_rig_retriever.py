"""Unit tests for the RigRetriever.

Author: Frank Dellaert
"""

import unittest
from pathlib import Path

from gtsfm.loader.hilti_loader import HiltiLoader
from gtsfm.retriever.rig_retriever import RigRetriever

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
TEST_DATASET_DIR_PATH = DATA_ROOT_PATH / "hilti_exp4_small"


class TestRigRetriever(unittest.TestCase):
    def test_rig_retriever(self) -> None:
        """Assert that we can parse a constraints file from the Hilti SLAM team and get constraints."""

        loader = HiltiLoader(TEST_DATASET_DIR_PATH)
        retriever = RigRetriever(threshold=30)

        pairs = retriever.get_image_pairs(loader=loader)
        # We know these to be the right values from setUp() method.
        self.assertEqual(len(pairs), 44)
        expected = [
            (0, 1),
            (0, 3),
            (0, 5),
            (0, 6),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 14),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 14),
            (2, 7),
            (2, 12),
            (3, 8),
            (3, 13),
            (4, 5),
            (4, 6),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 14),
            (5, 6),
            (5, 8),
            (5, 10),
            (5, 11),
            (5, 13),
            (5, 14),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 14),
            (7, 12),
            (8, 10),
            (8, 13),
            (9, 10),
            (9, 11),
            (9, 14),
            (10, 11),
            (10, 13),
            (11, 14),
        ]  # regression
        self.assertEqual(pairs, expected)


if __name__ == "__main__":
    unittest.main()
