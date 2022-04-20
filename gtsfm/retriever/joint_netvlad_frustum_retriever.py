"""Retriever that includes both sequential and retrieval links.

Authors: John Lambert
"""

from typing import List, Tuple

import dask

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase
from gtsfm.retriever.netvlad_retriever import NetVLADRetriever
from gtsfm.retriever.frustum_retriever import FrustumRetriever


logger = logger_utils.get_logger()


class JointNetVLADFrustumRetriever(RetrieverBase):
    """Note: this class contains no .run() method."""
    def __init__(self, num_matched: int, max_frame_lookahead: int) -> None:
        """
        Args:
            num_matched: number of K potential matches to provide per query. These are the top "K" matches per query.
            max_frame_lookahead: maximum number of consecutive frames to consider for matching/co-visibility.
        """
        self._similarity_retriever = NetVLADRetriever(num_matched=num_matched)
        self._frustum_retriever = FrustumRetriever(max_frame_lookahead=max_frame_lookahead, )

    def create_computation_graph(self, loader: LoaderBase) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader. The length of this loader will provide the total number of images
                for exhaustive global descriptor matching.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        sim_pairs = self._similarity_retriever.create_computation_graph(loader)
        frustum_pairs = self._frustum_retriever.create_computation_graph(loader)

        pairs = dask.delayed(self.aggregate_pairs)(sim_pairs=sim_pairs, frustum_pairs=frustum_pairs)
        return pairs

    def aggregate_pairs(
        self, sim_pairs: List[Tuple[int, int]], frustum_pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Aggregate all image pair indices from both similarity-based and camera frustum-based retrieval.

        Args:
            sim_pairs: image pairs (i1,i2) from similarity-based retrieval.
            frustum_pairs: image pairs (i1,i2) from camera frustum-based retrieval.

        Returns:
            pairs: unique pairs (i1,i2) representing union of the input sets.
        """
        pairs = list(set(sim_pairs).union(set(frustum_pairs)))
        logger.info("Found %d pairs from the NetVLAD + Frustum Retriever.", len(pairs))
        return pairs
