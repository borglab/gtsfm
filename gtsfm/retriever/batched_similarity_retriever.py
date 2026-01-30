"""Row-batched similarity retriever optimized for GPU execution.

This retriever computes exact inner-product similarity (equivalent to cosine
similarity for normalized descriptors) using a memory-efficient row-batched
approach. Unlike SimilarityRetriever which materializes the full N×N matrix
in blocks, this implementation processes row batches and immediately reduces
with top-k, achieving O(batch × N) memory instead of O(N²).

Authors: Kathir Gounder, [your collaborators]
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

import gtsfm.utils.logger as logger_utils
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()


class BatchedSimilarityRetriever(RetrieverBase):
    def __init__(
        self,
        num_matched: int,
        min_score: float = 0.1,
        batch_size: int = 1024,
    ) -> None:
        """
        Row-batched similarity retriever for large-scale image matching.

        Args:
            num_matched: Number of top matches to return per query image.
            min_score: Minimum similarity score threshold.
            batch_size: Number of query rows to process per batch. Tune based on
                GPU memory. Memory per batch ≈ batch_size × N × 4 bytes.
        """
        self._num_matched = num_matched
        self._min_score = min_score
        self._batch_size = batch_size
        self._latest_query_results: Optional[List[List[Tuple[int, float]]]] = None

    def __repr__(self) -> str:
        return f"""
        BatchedSimilarityRetriever:
            Num. frames matched: {self._num_matched}
            Minimum score: {self._min_score}
            Batch size: {self._batch_size}
            Device: {"cuda" if torch.cuda.is_available() else "cpu"}
        """

    def set_num_matched(self, n: int) -> None:
        """Set the number of matched frames."""
        self._num_matched = n

    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> VisibilityGraph:
        """Compute potential image pairs using batched similarity search.

        Args:
            global_descriptors: Normalized global descriptors (e.g., NetVLAD, MegaLoc).
            image_fnames: File names of the images.
            plots_output_dir: Directory to save diagnostic text files.

        Returns:
            List of (i1, i2) image pairs.
        """
        if global_descriptors is None or len(global_descriptors) == 0:
            raise ValueError("Global descriptors must be provided and non-empty")

        num_images = len(global_descriptors)
        descriptors = np.stack(global_descriptors).astype(np.float32)

        if not descriptors.flags['C_CONTIGUOUS']:
            descriptors = np.ascontiguousarray(descriptors)

        start_time = time.time()
        scores, indices = self._batched_search(descriptors)
        search_time = time.time() - start_time

        logger.info(
            "Batched similarity search: N=%d, D=%d, time=%.2fs",
            num_images, descriptors.shape[1], search_time
        )

        pairs, per_query_results = self._collect_pairs(scores, indices, num_images)
        self._latest_query_results = per_query_results

        logger.info("Found %d pairs using BatchedSimilarityRetriever.", len(pairs))

        if plots_output_dir:
            self.save_diagnostics(image_fnames, pairs, plots_output_dir)

        return pairs

    def _batched_search(self, descriptors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Row-batched similarity search with O(batch × N) memory.

        For each batch of query rows, computes similarity against all database
        images, masks invalid pairs (j <= i), and extracts top-k matches.

        Args:
            descriptors: (N, D) array of normalized descriptors.

        Returns:
            scores: (N, k) top-k similarity scores per query.
            indices: (N, k) indices of top-k matches per query.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        n = len(descriptors)
        k = min(self._num_matched, n - 1)

        desc_gpu = torch.from_numpy(descriptors).to(device)

        all_scores = torch.zeros(n, k, dtype=torch.float32)
        all_indices = torch.zeros(n, k, dtype=torch.int64)

        for i_start in range(0, n, self._batch_size):
            i_end = min(i_start + self._batch_size, n)
            batch_desc = desc_gpu[i_start:i_end]

            # (batch, N) similarity matrix for this batch
            sim_batch = batch_desc @ desc_gpu.T

            # Mask lower triangular + diagonal: for global row i, mask columns [0, i]
            # This enforces i < j and removes self-matches
            for b in range(sim_batch.shape[0]):
                global_i = i_start + b
                sim_batch[b, :global_i + 1] = float('-inf')

            # Apply score threshold
            if self._min_score is not None:
                sim_batch = sim_batch.masked_fill(sim_batch < self._min_score, float('-inf'))

            scores_batch, indices_batch = torch.topk(sim_batch, k=k, dim=1)

            all_scores[i_start:i_end] = scores_batch.cpu()
            all_indices[i_start:i_end] = indices_batch.cpu()

        return all_scores.numpy(), all_indices.numpy()

    def _collect_pairs(
        self, scores: np.ndarray, indices: np.ndarray, num_images: int
    ) -> Tuple[VisibilityGraph, List[List[Tuple[int, float]]]]:
        """Convert top-k results to pair list.

        Upper-triangular constraint already enforced during search.
        """
        pairs: List[Tuple[int, int]] = []
        per_query_results: List[List[Tuple[int, float]]] = []

        for i in range(num_images):
            query_matches: List[Tuple[int, float]] = []
            for k_idx in range(scores.shape[1]):
                j = int(indices[i, k_idx])
                score = float(scores[i, k_idx])

                if not np.isfinite(score) or j < 0:
                    continue

                pairs.append((i, j))
                query_matches.append((j, score))
            per_query_results.append(query_matches)

        return pairs, per_query_results

    def save_diagnostics(
        self,
        image_fnames: List[str],
        pairs: VisibilityGraph,
        plots_output_dir: Optional[Path],
    ) -> None:
        """Save retrieval diagnostics to text files.

        Note: Unlike SimilarityRetriever, we do NOT save a dense heatmap image
        since this retriever is designed for scales where N×N is prohibitive.
        """
        if plots_output_dir is None:
            return

        os.makedirs(plots_output_dir, exist_ok=True)

        # Save pair list
        pairs_path = plots_output_dir / "retrieved_pairs.txt"
        with open(pairs_path, "w") as f:
            f.write(f"# BatchedSimilarityRetriever Pairs\n")
            f.write(f"# Num Pairs: {len(pairs)}\n")
            f.write(f"# Min Score: {self._min_score}\n")
            f.write("# Format: Index1 Index2 Name1 Name2\n")
            for i, j in pairs:
                f.write(f"{i} {j} {image_fnames[i]} {image_fnames[j]}\n")
        logger.info("Saved pair list to %s", pairs_path)

        # Save ranked scores
        if self._latest_query_results is None:
            return

        ranked_path = plots_output_dir / "similarity_named_pairs.txt"
        with open(ranked_path, "w") as f:
            f.write("# Format: score name_i name_j\n")
            for i, matches in enumerate(self._latest_query_results):
                name_i = image_fnames[i]
                for j, score in matches:
                    f.write(f"{score:.4f} {name_i} {image_fnames[j]}\n")
        logger.info("Saved ranked pairs to %s", ranked_path)

        self._latest_query_results = None