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
    def run(self, loader: LoaderBase) -> List[Tuple[int, int]]:
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
            for i2 in range(num_images):
                if i1 >= i2:
                    continue
                if (i2 - i1) > self._num_matched:
                    continue
                pairs.append((i1, i2))

        logger.info("Found %d pairs from the SequentialRetriever", len(pairs))
        return pairs
