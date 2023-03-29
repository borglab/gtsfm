"""Sequential retriever, that uses a sliding window/fixed lookahead to propose image pairs.

Only useful for temporally ordered data.

Authors: John Lambert
"""
from typing import List, Tuple

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()


class SequentialRetriever(RetrieverBase):
    def __init__(self, max_frame_lookahead: int) -> None:
        """
        Args:
            max_frame_lookahead: maximum number of consecutive frames to consider for matching/co-visibility.
        """
        self._max_frame_lookahead = max_frame_lookahead

    def apply(self, loader: LoaderBase) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader. The length of this loader will provide the total number of images
                for exhaustive global descriptor matching.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        num_images = len(loader)

        pairs = []
        for i1 in range(num_images):
            max_i2 = min(i1 + self._max_frame_lookahead + 1, num_images)
            for i2 in range(i1 + 1, max_i2):
                pairs.append((i1, i2))

        logger.info("Found %d pairs from the SequentialRetriever", len(pairs))
        return pairs
