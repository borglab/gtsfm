"""Retriever that includes both sequential and retrieval links.

Authors: John Lambert
"""

from typing import List, Tuple

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase
from gtsfm.retriever.netvlad_retriever import NetVLADRetriever
from gtsfm.retriever.sequential_retriever import SequentialRetriever


logger = logger_utils.get_logger()


class JointNetVLADSequentialRetriever(RetrieverBase):

    def __init__(self, num_matched: int) -> None:
        """
        Args:
            num_matched: number of K potential matches to provide per query. These are the top "K" matches per query.
        """
        self._similarity_retriever = NetVLADRetriever(num_matched=num_matched)
        self._seq_retriever = SequentialRetriever(num_matched=num_matched)

    def run(self, loader: LoaderBase) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader. The length of this loader will provide the total number of images
                for exhaustive global descriptor matching.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        sim_pairs = self._similarity_retriever.run(loader)
        seq_pairs = self._seq_retriever.run(loader)

        pairs = list(set(sim_pairs).union(seq_pairs))
        logger.info(f"Found %d pairs from the NetVLAD + Sequential Retriever.", len(pairs))
        return pairs