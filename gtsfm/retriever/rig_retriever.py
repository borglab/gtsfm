"""Retriever for camera rigs that come with a constraints file.

Author: Frank Dellaert
"""

from pathlib import Path
from typing import List, Tuple
from gtsfm.loader.hilti_loader import HiltiLoader

from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase

INTRA_RIG_VALID_PAIRS = {(0, 1), (0, 3), (1, 4)}


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
        # Get between-rig constraints from HiltiLoader.
        assert isinstance(loader, HiltiLoader)
        constraints = loader.constraints

        # Get edges from those constraints.
        pairs = set(sum([c.edges(self._threshold) for c in constraints], []))

        # Add all intra-rig pairs even if no LIDAR signal.
        for rig_index in range(loader.max_rig_index):
            for c1, c2 in INTRA_RIG_VALID_PAIRS:
                pairs.add(
                    (loader.image_from_rig_and_camera(rig_index, c1), loader.image_from_rig_and_camera(rig_index, c2))
                )

        pairs = list(pairs)
        pairs.sort()
        return pairs
