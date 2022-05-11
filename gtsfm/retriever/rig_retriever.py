"""Retriever for camera rigs that come with a constraints file.

Author: Frank Dellaert
"""

from pathlib import Path
from typing import List, Tuple
from gtsfm.loader.hilti_loader import HiltiLoader

from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase


class RigRetriever(RetrieverBase):
    """Retriever for camera rigs inspired by the Hilti challenge."""

    def __init__(self, threshold: int = 100):
        """Create RigRetriever

        Args:
            threshold (int, optional): amount of "proxy" correspondences that will trigger an image-pair. Defaults to 100.
        """
        self._threshold = threshold

    def run(self, loader: LoaderBase) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        assert isinstance(loader, HiltiLoader)
        return sum([c.edges(self._threshold) for c in loader.constraints], [])
