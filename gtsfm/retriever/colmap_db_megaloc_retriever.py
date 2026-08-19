"""Hybrid retriever: keep each image's top-K COLMAP-verified neighbors, ranked by MegaLoc similarity.

Sparsifies COLMAP's dense vocab-tree view graph (e.g. ~140k pairs on St Peter's) down to a MegaLoc-
selected graph, WITHOUT fragmenting it: for every image we keep its ``num_matched`` most-MegaLoc-similar
neighbors *among the COLMAP-verified ones*. This is a restricted per-image top-K (not a strict
intersection of two independent top-K sets — that collapses when MegaLoc and COLMAP rank neighbors
differently, as they do on repetitive scenes). Every kept pair is COLMAP-verified, so correspondences
still come from the db (via ``ColmapCorrespondenceGenerator``); the graph stays connected and near-full
coverage while the downstream Metis cluster tree shrinks.

Authors: Kathirvel Gounder
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.colmap_db_retriever import ColmapDBRetriever
from gtsfm.retriever.retriever_base import RetrieverBase
from gtsfm.retriever.similarity_retriever import SimilarityRetriever, pairs_from_score_matrix

logger = logger_utils.get_logger()


class ColmapDBMegaLocRetriever(RetrieverBase):
    """Per-image top-K over COLMAP-verified neighbors, ranked by MegaLoc similarity."""

    def __init__(self, database_path: str, num_matched: int, min_score: float = 0.0) -> None:
        """
        Args:
            database_path: path to the COLMAP database.db (features + verified two-view geometries).
            num_matched: number of COLMAP-verified neighbors to keep per image, ranked by MegaLoc sim.
            min_score: minimum MegaLoc similarity to keep a (COLMAP-verified) neighbor. Default 0.0 =
                rank-only (COLMAP already gates geometric quality; MegaLoc just picks WHICH K to keep).
        """
        self._colmap_retriever = ColmapDBRetriever(database_path)
        self._similarity_retriever = SimilarityRetriever(num_matched=num_matched, min_score=min_score)
        self._num_matched = num_matched
        self._min_score = min_score

    def __repr__(self) -> str:
        return (
            "ColmapDBMegaLocRetriever:\n"
            f"    num_matched (COLMAP neighbors/image): {self._num_matched}\n"
            f"    min_score: {self._min_score}\n"
            f"    {self._colmap_retriever}"
        )

    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> VisibilityGraph:
        """Return each image's top-K most-MegaLoc-similar COLMAP-verified neighbors.

        Args:
            global_descriptors: MegaLoc descriptors, one per image (required for the ranking).
            image_fnames: file names of the images.
            plots_output_dir: directory to save plots to. If None, plots are not saved.

        Returns:
            Sparsified visibility graph: sorted (i, j) pairs with i < j.
        """
        colmap_pairs = self._colmap_retriever.get_image_pairs(
            global_descriptors=None, image_fnames=image_fnames, plots_output_dir=plots_output_dir
        )

        if global_descriptors is None:
            logger.warning(
                "🗄️🔎 ColmapDBMegaLocRetriever: no global descriptors provided; returning all %d COLMAP "
                "pairs unfiltered (set image_pairs_generator.global_descriptor to enable the MegaLoc filter).",
                len(colmap_pairs),
            )
            return colmap_pairs

        num_images = len(image_fnames)
        sim = self._similarity_retriever.compute_similarity_matrix(global_descriptors)

        # Candidate mask: only COLMAP-verified upper-triangular pairs are allowed. Everything else is
        # marked invalid so the per-image top-K picks from an image's COLMAP neighbors, not all images.
        is_invalid = ~np.triu(np.ones((num_images, num_images), dtype=bool))
        np.fill_diagonal(is_invalid, True)
        if colmap_pairs:
            ci = np.fromiter((i for i, _ in colmap_pairs), dtype=int, count=len(colmap_pairs))
            cj = np.fromiter((j for _, j in colmap_pairs), dtype=int, count=len(colmap_pairs))
            colmap_mask = np.zeros((num_images, num_images), dtype=bool)
            colmap_mask[ci, cj] = True  # COLMAP pairs are already i < j
            is_invalid |= ~colmap_mask

        pairs: VisibilityGraph = sorted(
            pairs_from_score_matrix(
                sim, invalid=is_invalid, num_select=self._num_matched, min_score=self._min_score
            )
        )
        logger.info(
            "🗄️🔎 ColmapDBMegaLocRetriever: %d COLMAP-verified pairs → top-%d per image by MegaLoc "
            "(min_score=%.2f) = %d pairs (from %d images).",
            len(colmap_pairs),
            self._num_matched,
            self._min_score,
            len(pairs),
            num_images,
        )
        return pairs
