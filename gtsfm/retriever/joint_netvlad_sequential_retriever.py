"""Retriever that includes both sequential and retrieval links.

Authors: John Lambert
"""

from typing import List, Tuple

import dask

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.netvlad_retriever import NetVLADRetriever
from gtsfm.retriever.retriever_base import RetrieverBase
from gtsfm.retriever.sequential_retriever import SequentialRetriever

logger = logger_utils.get_logger()


class JointNetVLADSequentialRetriever(RetrieverBase):
    """Note: this class contains no .run() method."""

    def __init__(self, num_matched: int, max_frame_lookahead: int) -> None:
        """
        Args:
            num_matched: number of K potential matches to provide per query. These are the top "K" matches per query.
            max_frame_lookahead: maximum number of consecutive frames to consider for matching/co-visibility.
        """
        self._num_matched = num_matched
        self._similarity_retriever = NetVLADRetriever(num_matched=num_matched)
        self._seq_retriever = SequentialRetriever(max_frame_lookahead=max_frame_lookahead)

    def create_computation_graph(self, loader: LoaderBase) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader. The length of this loader will provide the total number of images
                for exhaustive global descriptor matching.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        sim_pairs = self._similarity_retriever.create_computation_graph(loader)
        seq_pairs = self._seq_retriever.create_computation_graph(loader)

        pairs = dask.delayed(self.aggregate_pairs)(sim_pairs=sim_pairs, seq_pairs=seq_pairs)
        return pairs

    def aggregate_pairs(
        self, sim_pairs: List[Tuple[int, int]], seq_pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Aggregate all image pair indices from both similarity-based and sequential retrieval.

        Args:
            sim_pairs: image pairs (i1,i2) from similarity-based retrieval.
            seq_pairs: image pairs (i1,i2) from sequential retrieval.

        Returns:
            pairs: unique pairs (i1,i2) representing union of the input sets.
        """
        pairs = list(set(sim_pairs).union(set(seq_pairs)))
        logger.info("Found %d pairs from the NetVLAD + Sequential Retriever.", len(pairs))
        return pairs
