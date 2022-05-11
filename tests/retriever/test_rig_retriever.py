"""Unit tests for the RigRetriever.

Author: Frank Dellaert
"""

import tempfile
import unittest
from pathlib import Path
from typing import Optional

from gtsam import Cal3Bundler, Pose3
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.rig_retriever import RigRetriever

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
DEFAULT_FOLDER = DATA_ROOT_PATH / "hilti"


class HiltiLoader(LoaderBase):
    """Only reads constraints for now."""

    def __init__(self, root_path: Path):
        """Initialize with Hilti dataset directory."""
        self.constraints_file = root_path / "constraints.txt"

    def __len__(self) -> int:
        return 0

    def get_image_full_res(self, index: int) -> Image:
        return None

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3Bundler]:
        return None

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        return None


class TestRigRetriever(unittest.TestCase):
    def test_rig_retriever(self) -> None:
        """Assert that we can parse a constraints file from the Hilti SLAM team and get constraints."""

        loader = HiltiLoader(DEFAULT_FOLDER)
        constraints_path = DEFAULT_FOLDER / "test_constraints.txt"
        retriever = RigRetriever(constraints_path, threshold=30)

        pairs = retriever.run(loader=loader)

        # We know these to be the right values from setUp() method.
        self.assertEqual(len(pairs), 2)
        expected = [(5, 10), (6, 17)]
        self.assertEqual(pairs[:5], expected)


if __name__ == "__main__":
    unittest.main()
