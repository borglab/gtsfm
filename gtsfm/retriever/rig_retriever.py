"""Retriever for camera rigs that come with a constraints file.

Author: Frank Dellaert
"""

from pathlib import Path
from typing import List, Tuple

from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase
from gtsfm.common.constraint import Constraint


class RigRetriever(RetrieverBase):
    """Retriever for camera rigs inspired by the Hilti challenge."""

    def __init__(self, constraints_path: Path, threshold: int = 100):
        """Initialize with path to a constraints file."""
        self._constraints_path = constraints_path
        self._threshold = threshold

    def run(self, loader: LoaderBase) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        constraints = Constraint.read(str(self._constraints_path))

        return sum([c.edges(self._threshold) for c in constraints], [])
