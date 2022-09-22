"""Retriever for camera rigs that come with a constraints file.

Author: Frank Dellaert
"""

from typing import List, Tuple

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.hilti_loader import HiltiLoader
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()

INTRA_RIG_VALID_PAIRS = {(0, 1), (0, 3), (1, 4)}

CAM2_FRAME_LOOKAHEAD = 2  # number of frames to look ahead for cam2


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

        num_cam2_pairs = list(filter(lambda edge: edge[0] % 5 == 2 or edge[1] % 5 == 2, pairs))

        logger.info("Found %d pairs from the constraints file", len(pairs))
        logger.info("Found %d pairs with cam2 from the constraints file", len(num_cam2_pairs))

        # Add all intra-rig pairs even if no LIDAR signal.
        # Add all inter frames CAM2 pairs
        for rig_index in range(loader.num_rig_poses):
            for c1, c2 in INTRA_RIG_VALID_PAIRS:
                pairs.add(
                    (loader.image_from_rig_and_camera(rig_index, c1), loader.image_from_rig_and_camera(rig_index, c2))
                )
            for next_rig_idx in range(rig_index + 1, min(rig_index + 1 + CAM2_FRAME_LOOKAHEAD, loader.num_rig_poses)):
                pairs.add(
                    (loader.image_from_rig_and_camera(rig_index, 2), loader.image_from_rig_and_camera(next_rig_idx, 2))
                )

        pairs = list(pairs)
        pairs.sort()

        # check on pairs
        for i1, i2 in pairs:
            if i1 > i2:
                raise ValueError("Ordering not imposed on i1, i2")

        logger.info("Found %d pairs from the RigRetriever", len(pairs))
        return pairs
