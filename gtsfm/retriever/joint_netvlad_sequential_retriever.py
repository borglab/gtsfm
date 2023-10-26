"""Retriever that includes both sequential and retrieval links.

Authors: John Lambert
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.retriever.netvlad_retriever import NetVLADRetriever
from gtsfm.retriever.retriever_base import RetrieverBase, ImageMatchingRegime
from gtsfm.retriever.sequential_retriever import SequentialRetriever

logger = logger_utils.get_logger()


class JointNetVLADSequentialRetriever(RetrieverBase):
    """Retriever that includes both sequential and retrieval links."""

    def __init__(self, num_matched: int, min_score: float, max_frame_lookahead: int) -> None:
        """Initializes sub-retrievers.

        Args:
            num_matched: Number of K potential matches to provide per query. These are the top "K" matches per query.
            min_score: Minimum allowed similarity score to accept a match.
            max_frame_lookahead: Maximum number of consecutive frames to consider for matching/co-visibility.
        """
        super().__init__(matching_regime=ImageMatchingRegime.SEQUENTIAL_WITH_RETRIEVAL)
        self._num_matched = num_matched
        self._similarity_retriever = NetVLADRetriever(num_matched=num_matched, min_score=min_score)
        self._seq_retriever = SequentialRetriever(max_frame_lookahead=max_frame_lookahead)

    def __repr__(self) -> str:
        return f"""
        JointNetVLADSequentialRetriever:
            Similarity retriever: {self._similarity_retriever}
            Sequential retriever: {self._seq_retriever}
        """

    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            global_descriptors: the global descriptors for the retriever, if needed.
            image_fnames: file names of the images
            plots_output_dir: Directory to save plots to. If None, plots are not saved.

        Returns:
            List of (i1,i2) image pairs.
        """
        sim_pairs = self._similarity_retriever.get_image_pairs(
            global_descriptors=global_descriptors, image_fnames=image_fnames, plots_output_dir=plots_output_dir
        )
        seq_pairs = self._seq_retriever.get_image_pairs(
            global_descriptors=None, image_fnames=image_fnames, plots_output_dir=plots_output_dir
        )

        return self._aggregate_pairs(sim_pairs=sim_pairs, seq_pairs=seq_pairs)

    def _aggregate_pairs(
        self, sim_pairs: List[Tuple[int, int]], seq_pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Aggregate all image pair indices from both similarity-based and sequential retrieval.

        Args:
            sim_pairs: Image pairs (i1,i2) from similarity-based retrieval.
            seq_pairs: Image pairs (i1,i2) from sequential retrieval.

        Returns:
            Unique pairs (i1,i2) representing union of the input sets.
        """
        pairs = list(set(sim_pairs).union(set(seq_pairs)))
        logger.info("Found %d pairs from the NetVLAD + Sequential Retriever.", len(pairs))
        return pairs
