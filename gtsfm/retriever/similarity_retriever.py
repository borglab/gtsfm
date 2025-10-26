"""Retriever implementation which uses a global image descriptor to suggest potential image pairs.
Note: Similarity computation based off of Paul-Edouard Sarlin's HLOC:
Reference: https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/pairs_from_retrieval.py
https://openaccess.thecvf.com/content_cvpr_2016/papers/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.pdf
Authors: John Lambert
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

import gtsfm.utils.logger as logger_utils
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()
MAX_NUM_IMAGES = 10000


@dataclass
class SubBlockSimilarityResult:
    i_start: int
    i_end: int
    j_start: int
    j_end: int
    sub_block: torch.Tensor


class SimilarityRetriever(RetrieverBase):
    def __init__(self, num_matched: int, min_score: float = 0.1, blocksize: int = 50) -> None:
        """
        Args:
            num_matched: Number of K potential matches to provide per query. These are the top "K" matches per query.
            min_score: Minimum allowed similarity score to accept a match.
            blocksize: Size of matching sub-blocks when creating similarity matrix.
        """
        self._num_matched = num_matched
        self._blocksize = blocksize
        self._min_score = min_score

    def __repr__(self) -> str:
        return f"""
        SimilarityRetriever:
            Num. frames matched: {self._num_matched}
            Block size: {self._blocksize}
            Minimum score: {self._min_score}
        """

    def set_num_matched(self, n) -> None:
        """Set the number of matched frames for similarity matching."""
        self._num_matched = n

    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> VisibilityGraph:
        """Compute potential image pairs.

        Args:
            global_descriptors: the global descriptors for the retriever, if needed.
            image_fnames: file names of the images
            plots_output_dir: Directory to save plots to. If None, plots are not saved.

        Returns:
            List of (i1,i2) image pairs.
        """
        if global_descriptors is None:
            raise ValueError("Global descriptors need to be provided")
        sim = self.compute_similarity_matrix(global_descriptors)
        return self.compute_pairs_from_similarity_matrix(
            sim=sim, image_fnames=image_fnames, plots_output_dir=plots_output_dir
        )

    def compute_similarity_matrix(self, global_descriptors: List[np.ndarray]) -> torch.Tensor:
        """Compute a similarity matrix between all pairs of images.
        We use block matching, to avoid excessive memory usage.
        We cannot fit more than 50x50 sized block into memory, on a 16 GB RAM machine.
        A similar blocked exhaustive matching implementation can be found in COLMAP:
        https://github.com/colmap/colmap/blob/dev/src/feature/matching.cc#L899

        Args:
            global_descriptors: global descriptors, one per image.

        Returns:
            Similarity matrix as tensor of shape (num_images, num_images).
        """
        num_images = len(global_descriptors)
        if num_images > MAX_NUM_IMAGES:
            raise RuntimeError("Cannot construct similarity matrix of this size.")

        sub_block_results: List[SubBlockSimilarityResult] = []
        num_blocks = math.ceil(num_images / self._blocksize)

        # TODO(Ayush, John): do we still need to do sub_block computations?
        for block_i in range(num_blocks):
            for block_j in range(block_i, num_blocks):
                sub_block_results.append(
                    self._compute_similarity_sub_block(
                        global_descriptors=global_descriptors, block_i=block_i, block_j=block_j
                    )
                )

        sim = self._aggregate_sub_blocks(sub_block_results=sub_block_results, num_images=num_images)
        return sim

    def _compute_similarity_sub_block(self, global_descriptors: List[np.ndarray], block_i: int, block_j: int):
        """Compute a sub-block of an global descriptor based similarity matrix.
        Args:
            global_descriptors: global descriptors, one per image.
            block_i: row index of sub-block.
            block_j: column index of sub-block.
        Returns:
            sub-block similarity result.
        """
        num_images = len(global_descriptors)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_blocks = math.ceil(num_images / self._blocksize)
        # only compute the upper triangular portion of the similarity matrix.
        logger.info("Computing matching block (%d/%d,%d/%d)", block_i, num_blocks - 1, block_j, num_blocks - 1)
        i_start = block_i * self._blocksize
        i_end = (block_i + 1) * self._blocksize
        i_end = min(i_end, num_images)
        j_start = block_j * self._blocksize
        j_end = (block_j + 1) * self._blocksize
        j_end = min(j_end, num_images)
        # Form (K,D) for K images.
        block_i_query_descriptors = torch.from_numpy(np.array(global_descriptors[i_start:i_end]))
        block_j_query_descriptors = torch.from_numpy(np.array(global_descriptors[j_start:j_end]))
        # Einsum equivalent to (img_descs @ img_descs.T)
        sim_block = torch.einsum(
            "id,jd->ij", block_i_query_descriptors.to(device), block_j_query_descriptors.to(device)
        )
        return SubBlockSimilarityResult(i_start=i_start, i_end=i_end, j_start=j_start, j_end=j_end, sub_block=sim_block)

    def _aggregate_sub_blocks(self, sub_block_results: List[SubBlockSimilarityResult], num_images: int) -> torch.Tensor:
        """Aggregate results from many independently computed sub-blocks of the similarity matrix into a single matrix.

        Args:
            sub_block_results: Metadata and results of similarity matrix sub-block computation.
            num_images: Number of images to compare for matching.

        Returns:
            sim: Tensor of shape (num_images, num_images) representing similarity matrix.
        """
        sim = torch.zeros((num_images, num_images))
        for sr in sub_block_results:
            sim[sr.i_start : sr.i_end, sr.j_start : sr.j_end] = sr.sub_block
        return sim

    def compute_pairs_from_similarity_matrix(
        self, sim: torch.Tensor, image_fnames: List[str], plots_output_dir: Optional[Path] = None
    ) -> VisibilityGraph:
        """

        Args:
            sim: Tensor of shape (num_images, num_images) representing similarity matrix.
            image_fnames: the names of the images
            plots_output_dir: Directory to save plots to. If None, plots are not saved.

        Returns:
            pair_indices: (i1,i2) image pairs.
        """
        num_images = len(image_fnames)
        # Avoid self-matching and disallow lower triangular portion
        is_invalid_mat = ~np.triu(np.ones((num_images, num_images), dtype=bool))
        np.fill_diagonal(a=is_invalid_mat, val=True)
        pairs = pairs_from_score_matrix(
            sim, invalid=is_invalid_mat, num_select=self._num_matched, min_score=self._min_score
        )
        named_pairs = [(image_fnames[i], image_fnames[j]) for i, j in pairs]
        if plots_output_dir:
            os.makedirs(plots_output_dir, exist_ok=True)
            # Save image of similarity matrix.
            plt.imshow(np.triu(sim.detach().cpu().numpy()))
            plt.title("Image Similarity Matrix")
            plt.savefig(str(plots_output_dir / "similarity_matrix.jpg"), dpi=500)
            plt.close("all")
            # Save values in similarity matrix.
            np.savetxt(
                fname=str(plots_output_dir / "similarity_matrix.txt"),
                X=sim.detach().cpu().numpy(),
                fmt="%.2f",
                delimiter=",",
            )

            # Save named pairs and scores.
            with open(plots_output_dir / "netvlad_named_pairs.txt", "w") as fid:
                for _named_pair, _pair_ind in zip(named_pairs, pairs):
                    fid.write("%.4f %s %s\n" % (sim[_pair_ind[0], _pair_ind[1]], _named_pair[0], _named_pair[1]))

        logger.info("Found %d pairs from the NetVLAD Retriever.", len(pairs))
        return pairs


def pairs_from_score_matrix(
    scores: torch.Tensor, invalid: np.ndarray, num_select: int, min_score: Optional[float] = None
) -> VisibilityGraph:
    """Identify image pairs from a score matrix.

    Note: Similarity computation here is based off of Paul-Edouard Sarlin's HLOC:
    Reference: https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/pairs_from_retrieval.py

    Args:
        scores: (K1,K2) for matching K1 images against K2 images.
        invalid: (K1,K2) boolean array indicating invalid match pairs (e.g. self-matches).
        num_select: Number of matches to select, per query.
        min_score: Minimum allowed similarity score.

    Returns:
        pairs: Tuples representing pairs (i1,i2) of images.
    """
    N = scores.shape[0]
    # if there are only N images to choose from, selecting more than N is not allowed
    num_select = min(num_select, N)
    assert scores.shape == invalid.shape
    invalid = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        # logical OR.
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float("-inf"))
    topk = torch.topk(scores, k=num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()
    pairs = []
    for i_raw, j_raw in zip(*np.where(valid)):
        j = int(indices[i_raw, j_raw])
        pairs.append((int(i_raw), j))
    return pairs
