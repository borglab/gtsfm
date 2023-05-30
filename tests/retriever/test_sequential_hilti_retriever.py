"""Unit tests for the SequentialHiltiRetriever.

Author: John Lambert
"""

import unittest
from pathlib import Path

from gtsfm.loader.hilti_loader import HiltiLoader
from gtsfm.retriever.sequential_hilti_retriever import SequentialHiltiRetriever

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
TEST_DATASET_DIR_PATH = DATA_ROOT_PATH / "hilti_exp4_small"


class TestSequentialHiltiRetriever(unittest.TestCase):
    def test_sequential_retriever(self) -> None:
        """Assert that we get 30 total matches with a lookahead of 3 frames on the Door Dataset."""

        loader = HiltiLoader(TEST_DATASET_DIR_PATH)
        max_frame_lookahead = 3
        retriever = SequentialHiltiRetriever(max_frame_lookahead=max_frame_lookahead)
        pairs = retriever.apply(loader=loader)
        self.assertEqual(len(pairs), 42)  # regression, did not check carefully
        expected_pairs = [
            (0, 1),
            (0, 3),
            (0, 5),
            (0, 6),
            (0, 8),
            (0, 10),
            (0, 11),
            (0, 13),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 14),
            (2, 7),
            (2, 12),
            (3, 5),
            (3, 8),
            (3, 10),
            (3, 13),
            (4, 6),
            (4, 9),
            (4, 11),
            (4, 14),
            (5, 6),
            (5, 8),
            (5, 10),
            (5, 11),
            (5, 13),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 14),
            (7, 12),
            (8, 10),
            (8, 13),
            (9, 11),
            (9, 14),
            (10, 11),
            (10, 13),
            (11, 14),
        ]  # regression
        self.assertEqual(pairs, expected_pairs)


if __name__ == "__main__":
    unittest.main()
