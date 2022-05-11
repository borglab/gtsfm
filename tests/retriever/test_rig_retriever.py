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

        pairs = retriever.run(loader=loader)

        # We know these to be the right values from setUp() method.
        self.assertEqual(len(pairs), 32)
        expected = [
            (5, 10),
            (5, 11),
            (5, 13),
            (5, 14),
            (6, 10),
            (6, 11),
            (6, 14),
            (8, 10),
            (8, 13),
            (9, 10),
            (9, 11),
            (9, 14),
            (0, 10),
            (0, 11),
            (0, 14),
            (1, 10),
            (1, 11),
            (1, 14),
            (3, 13),
            (4, 10),
            (4, 11),
            (4, 14),
            (0, 5),
            (0, 6),
            (0, 9),
            (1, 5),
            (1, 6),
            (1, 9),
            (3, 8),
            (4, 5),
            (4, 6),
            (4, 9),
        ]
        self.assertEqual(pairs, expected)


if __name__ == "__main__":
    unittest.main()
