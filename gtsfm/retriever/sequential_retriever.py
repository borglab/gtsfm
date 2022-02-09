"""Sequential retriever, that uses a sliding window/fixed lookahead to propose image pairs.

Only useful for temporally ordered data.

Authors: John Lambert
"""
from typing import List, Tuple

from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase


class SequentialRetriever(RetrieverBase):
    def run(self, loader: LoaderBase, num_matched: int = 2) -> List[Tuple[int, int]]:
        """
        Args:
            loader: image loader.
            num_images: total number of images for exhaustive global descriptor matching.
            num_matched: number of K potential matches to provide per query. These are the top "K" matches per query.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        num_images = len(loader)

        pairs = []
        for i1 in range(num_images):
            for i2 in range(num_images):
                if i1 >= i2:
                    continue
                if (i2 - i1) > num_matched:
                    continue
                pairs.append((i1, i2))

        return pairs
