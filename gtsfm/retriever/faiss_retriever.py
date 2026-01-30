import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Attempt to import faiss; handle cases where it might not be installed if strict deps aren't enforced
try:
    import faiss
except ImportError:
    faiss = None

import gtsfm.utils.logger as logger_utils
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()

class FaissRetriever(RetrieverBase):
    def __init__(
        self,
        num_matched: int,
        min_score: float = 0.1,
        index_type: str = "flat",
        nlist: Optional[int] = None,
        nprobe: Optional[int] = None,
        hnsw_m: int = 32,
        hnsw_ef_search: int = 64,
        hnsw_ef_construction: int = 200,
        oversample_factor: float = 1.0,
    ) -> None:
        """
        Retriever implementation using FAISS (Facebook AI Similarity Search) for efficient 
        Approximate Nearest Neighbor (ANN) search.

        This class uses IndexFlatIP (Inner Product). It assumes input descriptors are 
        already normalized (e.g. MegaLoc, NetVLAD), in which case Inner Product is 
        equivalent to Cosine Similarity.

        Args:
            num_matched: Number of K potential matches to provide per query.
            min_score: Minimum allowed similarity score to accept a match.
            index_type: FAISS index type: "flat", "ivf_flat", or "hnsw".
            nlist: IVF cluster count. If None, choose based on dataset size.
            nprobe: IVF probes per query. If None, choose based on nlist.
            hnsw_m: HNSW graph degree.
            hnsw_ef_search: HNSW search expansion factor.
            hnsw_ef_construction: HNSW construction expansion factor.
            oversample_factor: Multiplier for K when searching (>= 1.0) to recover
                more upper-triangular neighbors after filtering.
        """
        if faiss is None:
            raise ImportError("The 'faiss' library is required for FaissRetriever. Please install 'faiss-gpu' or 'faiss-cpu'.")

        self._num_matched = num_matched
        self._min_score = min_score

        self._index_type = self._normalize_index_type(index_type)
        self._nlist = nlist
        self._nprobe = nprobe
        self._hnsw_m = hnsw_m
        self._hnsw_ef_search = hnsw_ef_search
        self._hnsw_ef_construction = hnsw_ef_construction

        if oversample_factor < 1.0:
            logger.warning("oversample_factor must be >= 1.0; clamping to 1.0.")
        self._oversample_factor = max(1.0, oversample_factor)

        self._validate_index_type()
        self._latest_query_results: Optional[List[List[Tuple[int, float]]]] = None

    def __repr__(self) -> str:
        return f"""
        FaissRetriever:
            Num. frames matched: {self._num_matched}
            Minimum score: {self._min_score}
            Index type: {self._index_type}
            IVF nlist: {self._nlist if self._nlist is not None else "auto"}
            IVF nprobe: {self._nprobe if self._nprobe is not None else "auto"}
            HNSW M: {self._hnsw_m}
            HNSW efSearch: {self._hnsw_ef_search}
            HNSW efConstruction: {self._hnsw_ef_construction}
            Oversample factor: {self._oversample_factor}
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
        """Compute potential image pairs using FAISS.

        Args:
            global_descriptors: The global descriptors for the retriever.
            image_fnames: File names of the images.
            plots_output_dir: Directory to save diagnostic text files.

        Returns:
            List of (i1, i2) image pairs.
        """
        if global_descriptors is None or len(global_descriptors) == 0:
            raise ValueError("Global descriptors need to be provided and non-empty")

        num_images = len(global_descriptors)
        
        # 1. Prepare Data
        # FAISS expects float32 (C-contiguous) numpy arrays.
        # We assume descriptors are ALREADY normalized (e.g. MegaLoc), following the convention 
        # of the original SimilarityRetriever.
        descriptors = np.stack(global_descriptors).astype('float32')
        
        # If the array is not C-contiguous, FAISS might copy it internally, so we enforce it here.
        if not descriptors.flags['C_CONTIGUOUS']:
            descriptors = np.ascontiguousarray(descriptors)

        dim = descriptors.shape[1]

        # 2. Build Index
        start_time = time.time()
        index = self._build_index(descriptors)
        total_time = time.time() - start_time

        index.add(descriptors)
        logger.info("Built FAISS index for %d images (Dimension: %d) in %0.4f seconds", num_images, dim, total_time)

        # 3. Perform Search
        # We query (K * oversample_factor + 1) to recover upper-triangular neighbors.
        k_search = self._compute_k_search(num_images)
        
        # scores: (N, K), indices: (N, K)
        scores, indices = index.search(descriptors, k_search)

        # 4. Filter and Format Pairs
        start_time = time.time()
        pairs, per_query_results = self._filter_results(scores, indices, num_images)
        self._latest_query_results = per_query_results
        total_time = time.time() - start_time
        logger.info("Total %0.5f Time to compute similarity matrix", total_time)
        
        logger.info("Found %d pairs using FaissRetriever.", len(pairs))

        # 5. Save Diagnostics (Text only, as dense heatmaps are too large for this scale)
        if plots_output_dir:
            self.save_diagnostics(image_fnames, pairs, plots_output_dir)

        return pairs

    def _filter_results(
        self, scores: np.ndarray, indices: np.ndarray, num_images: int
    ) -> tuple[VisibilityGraph, List[List[Tuple[int, float]]]]:
        """Filter raw FAISS results into unique, valid pairs and per-query ranked matches.
        
        Criteria:
        1. Remove self-matches (i == j).
        2. Enforce Upper Triangular (i < j) to prevent duplicate pairs like (0,1) and (1,0).
        3. Enforce min_score threshold.
        """
        # Ensure we are working with CPU numpy arrays for iteration
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()
            indices = indices.cpu().numpy()
        elif isinstance(scores, np.ndarray) is False:
             # Handle generic array-like objects if necessary, though FAISS returns np or torch
            scores = np.array(scores)
            indices = np.array(indices)

        pairs: List[Tuple[int, int]] = []
        per_query_results: List[List[Tuple[int, float]]] = []

        # Iterate through queries
        for i in range(num_images):
            row_pairs = 0
            query_matches: List[Tuple[int, float]] = []
            for k in range(scores.shape[1]):
                j = int(indices[i, k])
                score = float(scores[i, k])

                # FAISS returns -1 if fewer than k_search neighbors exist (rare for IndexFlatIP)
                if j < 0: 
                    continue

                # 1. Enforce Upper Triangular (i < j)
                # This explicitly handles self-matches (where i == j) and prevents 
                # symmetric duplicates (checking A vs B and B vs A).
                if i >= j:
                    continue

                # 2. Enforce Score Threshold
                if score < self._min_score:
                    break

                pairs.append((i, j))
                query_matches.append((j, score))
                row_pairs += 1
                if row_pairs >= self._num_matched:
                    break
            per_query_results.append(query_matches)

        return pairs, per_query_results

    def save_diagnostics(
        self, 
        image_fnames: List[str], 
        pairs: VisibilityGraph, 
        plots_output_dir: Optional[Path]
    ) -> None:
        """Saves a text file of the retrieved pairs. 
        
        Unlike SimilarityRetriever, we do NOT save a dense heatmap image, 
        as this class is intended for scales where N*N is too large for memory.
        """
        if plots_output_dir is None:
            return

        os.makedirs(plots_output_dir, exist_ok=True)
        named_pairs_path = plots_output_dir / "retrieved_pairs.txt"
        
        with open(named_pairs_path, "w") as f:
            f.write(f"# FaissRetriever Pairs\n")
            f.write(f"# Num Pairs: {len(pairs)}\n")
            f.write(f"# Min Score: {self._min_score}\n")
            f.write("# Format: Index1 Index2 Name1 Name2\n")
            for (i, j) in pairs:
                name_i = image_fnames[i]
                name_j = image_fnames[j]
                f.write(f"{i} {j} {name_i} {name_j}\n")
        
        logger.info("Saved retrieval diagnostics to %s", named_pairs_path)

        if self._latest_query_results is None:
            logger.warning("No cached FAISS query results available to save scores.")
            return

        ranked_pairs_path = plots_output_dir / "faiss_named_pairs.txt"
        with open(ranked_pairs_path, "w") as f:
            f.write("# Format: score name_i name_j\n")
            for i, matches in enumerate(self._latest_query_results):
                name_i = image_fnames[i]
                for j, score in matches:
                    name_j = image_fnames[j]
                    f.write(f"{score:.4f} {name_i} {name_j}\n")
        logger.info("Saved ranked retrieval diagnostics to %s", ranked_pairs_path)
        self._latest_query_results = None

    def _normalize_index_type(self, index_type: str) -> str:
        normalized = index_type.strip().lower()
        if normalized in {"ivf", "ivfflat"}:
            return "ivf_flat"
        if normalized in {"hnsw_flat"}:
            return "hnsw"
        return normalized

    def _validate_index_type(self) -> None:
        valid = {"flat", "ivf_flat", "hnsw"}
        if self._index_type not in valid:
            raise ValueError(f"Unsupported FAISS index_type: {self._index_type}")

    def _resolve_nlist(self, num_images: int) -> int:
        if self._nlist is not None:
            return max(1, min(self._nlist, num_images))
        return max(1, min(int(4 * math.sqrt(num_images)), num_images))

    def _resolve_nprobe(self, nlist: int) -> int:
        if self._nprobe is not None:
            return max(1, min(self._nprobe, nlist))
        return max(1, min(32, max(1, nlist // 10)))

    def _build_index(self, descriptors: np.ndarray) -> "faiss.Index":
        num_images, dim = descriptors.shape
        if self._index_type == "flat":
            # IndexFlatIP implements exact search using Inner Product (Dot Product).
            # If vectors are normalized, A . B = Cosine Similarity.
            index = faiss.IndexFlatIP(dim)
        elif self._index_type == "ivf_flat":
            nlist = self._resolve_nlist(num_images)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(descriptors)
            nprobe = self._resolve_nprobe(nlist)
            index.nprobe = nprobe
            logger.info("FAISS IVF params: nlist=%d nprobe=%d", nlist, nprobe)
        elif self._index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, self._hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = self._hnsw_ef_search
            index.hnsw.efConstruction = self._hnsw_ef_construction
            logger.info(
                "FAISS HNSW params: M=%d efSearch=%d efConstruction=%d",
                self._hnsw_m,
                self._hnsw_ef_search,
                self._hnsw_ef_construction,
            )
        else:
            raise ValueError(f"Unsupported FAISS index_type: {self._index_type}")

        return index

    def _compute_k_search(self, num_images: int) -> int:
        k_search = int(self._num_matched * self._oversample_factor) + 1
        k_search = max(self._num_matched + 1, k_search)
        return min(num_images, k_search)
