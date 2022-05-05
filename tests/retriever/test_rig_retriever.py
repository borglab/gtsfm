"""Unit tests for the RigRetriever.

Author: Frank Dellaert
"""

import unittest
from pathlib import Path
from typing import List, Optional, Tuple

from gtsam import Cal3Bundler, Pose3
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase

# from gtsfm.retriever.rig_retriever import RigRetriever

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
DEFAULT_FOLDER = DATA_ROOT_PATH / "hilti"


class DummyLoader(LoaderBase):
    """Dummy loader for now."""

    def __len__(self) -> int:
        return 0

    def get_image_full_res(self, index: int) -> Image:
        return None

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3Bundler]:
        return None

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        return None


class RigRetriever(RetrieverBase):
    """Retriever for camera rigs inspired by the Hilti challenge."""

    def __init__(self, constraints_file: str):
        """Create with filename of constraints between rigs."""
        self._constraints_file = constraints_file

    def run(self, loader: LoaderBase) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        return []


class TestRigRetriever(unittest.TestCase):
    def test_rig_retriever(self) -> None:
        """Assert that we can parse a constraints file from the Hilti SLAM team and get edges."""

        loader = DummyLoader()
        retriever = RigRetriever(str(DEFAULT_FOLDER / "constraints.txt"))

        pairs = retriever.run(loader=loader)
        self.assertEqual(len(pairs), 9)

        expected_pairs = [
            (0, 1),
        ]
        self.assertEqual(pairs, expected_pairs)


if __name__ == "__main__":
    unittest.main()
