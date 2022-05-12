"""Retriever for camera rigs that come with a constraints file.

Author: Frank Dellaert
"""

from typing import List, Tuple
from gtsfm.loader.hilti_loader import HiltiLoader

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()

INTRA_RIG_VALID_PAIRS = {(0, 1), (0, 3), (1, 4)}


class RigRetriever(RetrieverBase):
    """Retriever for camera rigs inspired by the Hilti challenge."""

    def __init__(self, threshold: int = 100):
        """Create RigRetriever

        Args:
            threshold (int, optional): amount of "proxy" correspondences that will trigger an image-pair. Default 100.
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

        # Get pairs from those constraints.
        pairs = set(sum([c.predicted_pairs(self._threshold) for c in constraints], []))

        # Add all intra-rig pairs even if no LIDAR signal.
        for rig_index in range(loader.max_rig_index):
            for c1, c2 in INTRA_RIG_VALID_PAIRS:
                pairs.add(
                    (loader.image_from_rig_and_camera(rig_index, c1), loader.image_from_rig_and_camera(rig_index, c2))
                )

        pairs = list(pairs)
        pairs.sort()
        logger.info("Found %d pairs from the RigRetriever", len(pairs))
        return pairs
