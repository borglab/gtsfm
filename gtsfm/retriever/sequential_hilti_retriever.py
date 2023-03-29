"""Sequential retriever, that uses a sliding window/fixed lookahead and hardcoded rig structure to propose image pairs.

Authors: Ayush Baid.
"""
from typing import List, Tuple

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.hilti_loader import HiltiLoader
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()

INTRA_RIG_VALID_PAIRS = {(0, 1), (0, 3), (1, 4)}
INTER_RIG_VALID_PAIRS = {(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 4), (2, 2), (3, 0), (3, 3), (4, 1), (4, 4)}


class SequentialHiltiRetriever(RetrieverBase):
    """Temporary class for the hilti loader."""

    def __init__(self, max_frame_lookahead: int) -> None:
        """
        Args:
            max_frame_lookahead: maximum number of consecutive rig frames to consider for matching/co-visibility.
        """
        self._max_frame_lookahead = max_frame_lookahead

    def is_valid_pair(self, loader, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair. idx1 < idx2 is required.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """
        if idx1 >= idx2:
            return False

        rig_idx_i1 = loader.rig_from_image(idx1)
        rig_idx_i2 = loader.rig_from_image(idx2)

        cam_idx_i1 = loader.camera_from_image(idx1)
        cam_idx_i2 = loader.camera_from_image(idx2)
        if rig_idx_i1 == rig_idx_i2:
            return (cam_idx_i1, cam_idx_i2) in INTRA_RIG_VALID_PAIRS
        elif rig_idx_i1 < rig_idx_i2 and rig_idx_i2 - rig_idx_i1 <= self._max_frame_lookahead:
            return (cam_idx_i1, cam_idx_i2) in INTER_RIG_VALID_PAIRS

    def apply(self, loader: HiltiLoader) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader. The length of this loader will provide the total number of images
                for exhaustive global descriptor matching.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        pairs = []

        num_images = len(loader)
        for idx1 in range(num_images):
            for idx2 in range(num_images):
                if self.is_valid_pair(loader, idx1, idx2):
                    pairs.append((idx1, idx2))

        logger.info("Found %d pairs from the SequentialHiltiRetriever", len(pairs))
        return pairs
