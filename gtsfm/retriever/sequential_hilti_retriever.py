"""Sequential retriever, that uses a sliding window/fixed lookahead and hardcoded rig structure to propose image pairs.

Authors: Ayush Baid.
"""
from typing import List, Tuple

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.hilti_loader import HiltiLoader
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()


class SequentialHiltiRetriever(RetrieverBase):
    """Temporary class for the hilti loader."""

    def __init__(self, max_frame_lookahead: int) -> None:
        """
        Args:
            max_frame_lookahead: maximum number of consecutive frames to consider for matching/co-visibility.
        """
        self._max_frame_lookahead = max_frame_lookahead

    def run(self, loader: HiltiLoader) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader. The length of this loader will provide the total number of images
                for exhaustive global descriptor matching.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        pairs = loader.get_valid_pairs()  # TODO: move this logic here.

        logger.info("Found %d pairs from the SequentialHiltiRetriever", len(pairs))
        return pairs
