""" Tests for main driver class.

Authors: Frank Dellaert and Ayush Baid
"""

import sys
import unittest
from pathlib import Path

import dask

# TODO(dellaert): this is terrible!
PROJECT_DIR = Path(__file__).parents[1]
print(PROJECT_DIR)
sys.path.append(str(PROJECT_DIR))

from loader.folder_loader import FolderLoader


class FrontEndBase:
    """Base class for FrontEnd classes."""
    def __init__(self, loader):
        pass

    def create_compute_graph(self):
        return []

class DummyFrontEnd(FrontEndBase):
    """Dummy class to test dask compute graphs."""
    pass

class Result:
    """A class that contains all GTSFM results and diagnostics."""


class GTSFM:
    """Main GTSFM driver class."""

    def __init__(self, folder: str):
        """Simple constructor that just takes an input folder."""
        self._loader = FolderLoader(folder)
        self._frontend = DummyFrontEnd(self._loader)

    def run(self):
        """Run all stages and returns final result.

        Returns:
            Result -- A class that contains all results and diagnostics.
        """
        # Build the pipeline
        frontend_graph = self._frontend.create_compute_graph()

        matches = dask.compute(frontend_graph)

        return Result()


class TestGTSFM(unittest.TestCase):
    """Main test class for GTSFM."""

    def test_main(self):
        """Test the default invocation
        """
        sfm = GTSFM("sample_data")
        result = sfm.run()
        self.assertIsInstance(result, Result)


if __name__ == '__main__':
    unittest.main()
