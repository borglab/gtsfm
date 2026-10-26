"""Two way (mutual nearest neighbor) matcher.

Ref: https://github.com/colmap/colmap/blob/2b7230679957e4dccd590ab467931d6cfffb9ede/src/feature/sift.cc

Authors: Ayush Baid
"""
from enum import Enum
from typing import Dict, Optional, Tuple

import cv2 as cv
import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.matcher_base import MatcherBase


class MatchingDistanceType(Enum):
    """Type of distance metric to use for matching descriptors."""

    HAMMING = 1
    EUCLIDEAN = 2


class TwoWayMatcher(MatcherBase):
    """Two way (mutual nearest neighbor) matcher using OpenCV, with optional ratio test"""

    def __init__(
        self,
        distance_type: MatchingDistanceType = MatchingDistanceType.EUCLIDEAN,
        ratio_test_threshold: Optional[float] = None,
        search_method: str = "flann",
        flann_trees: int = 4,
        flann_checks: int = 256,
    ):
        """Initialize the matcher.

        Args:
            distance_type: distance type for matching.
            ratio_test_threshold: ratio test threshold (optional). Defaults to None (no ratio test applied).
            search_method: Defaults to "flann" for approximate KD-tree search of floating-point
                Euclidean descriptors such as SIFT. Use "brute_force" for exact search or binary descriptors.
            flann_trees: Number of randomized KD-trees (default 4). More trees increase build time and memory.
            flann_checks: Search effort per descriptor (default 256). Higher values trade speed for recall.
        """
        super().__init__()
        if search_method not in {"brute_force", "flann"}:
            raise ValueError("search_method must be 'brute_force' or 'flann'")
        if search_method == "flann" and distance_type is not MatchingDistanceType.EUCLIDEAN:
            raise ValueError("FLANN KD-tree search requires Euclidean descriptors; use brute_force for Hamming")
        if flann_trees < 1 or flann_checks < 1:
            raise ValueError("flann_trees and flann_checks must be positive")
        self._distance_type = distance_type
        self._ratio_test_threshold: Optional[float] = ratio_test_threshold
        self._search_method = search_method
        self._flann_trees = flann_trees
        self._flann_checks = flann_checks

    @property
    def cache_key(self) -> str:
        """Separate approximate results and search settings while retaining existing brute-force caches."""
        if getattr(self, "_search_method", "brute_force") == "brute_force":
            return type(self).__name__
        return (
            f"{type(self).__name__}_flann_trees{self._flann_trees}_checks{self._flann_checks}"
            f"_ratio{self._ratio_test_threshold}"
        )

    def match(
        self,
        keypoints_i1: Keypoints,  # pylint: disable=unused-argument
        keypoints_i2: Keypoints,  # pylint: disable=unused-argument
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int, int],  # pylint: disable=unused-argument
        im_shape_i2: Tuple[int, int, int],  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """Match descriptor vectors.

        Output format:
        1. Each row represents a match.
        2. First column represents keypoint index from image #i1.
        3. Second column represents keypoint index from image #i2.
        4. Matches are sorted in descending order of the confidence (score), if possible.

        Args:
            keypoints_i1: keypoints for image #i1, of length N1.
            keypoints_i2: keypoints for image #i2, of length N2.
            descriptors_i1: descriptors corr. to keypoints_i1.
            descriptors_i2: descriptors corr. to keypoints_i2.
            im_shape_i1: shape of image #i1, as height, width, channels.
            im_shape_i2: shape of image #i2, as height, width, channels.

        Returns:
            Match indices (sorted by confidence), as matrix of shape (N, 2), where N < min(N1, N2).
        """

        if descriptors_i1.size == 0 or descriptors_i2.size == 0:
            return np.array([])

        # OpenCV cannot index non-finite descriptors. Retain indices for remapping the results.
        valid_idx_i1 = np.nonzero(np.isfinite(descriptors_i1).all(axis=1))[0]
        valid_idx_i2 = np.nonzero(np.isfinite(descriptors_i2).all(axis=1))[0]
        min_descriptors = 2 if self._ratio_test_threshold is not None else 1
        if len(valid_idx_i1) < min_descriptors or len(valid_idx_i2) < min_descriptors:
            return np.empty((0, 2), dtype=np.uint32)

        if getattr(self, "_search_method", "brute_force") == "flann":
            if descriptors_i1.dtype.kind != "f" or descriptors_i2.dtype.kind != "f":
                raise ValueError("FLANN KD-tree search requires floating-point descriptors")
            descriptors_i1 = np.ascontiguousarray(descriptors_i1, dtype=np.float32)
            descriptors_i2 = np.ascontiguousarray(descriptors_i2, dtype=np.float32)

        match_indices = self.__perform_matching(descriptors_i1[valid_idx_i1], descriptors_i2[valid_idx_i2])

        if match_indices.size == 0:
            return np.array([])

        # remap them back
        match_indices[:, 0] = valid_idx_i1[match_indices[:, 0]]
        match_indices[:, 1] = valid_idx_i2[match_indices[:, 1]]

        return match_indices

    def __init_opencv_matcher(self) -> cv.DescriptorMatcher:
        """Initialize the OpenCV matcher.

        Returns:
            Matcher object.
        """
        if getattr(self, "_search_method", "brute_force") == "flann":
            # Each direction/call owns its index, so threaded pair matching shares no mutable FLANN state.
            return cv.FlannBasedMatcher(
                dict(algorithm=1, trees=self._flann_trees), dict(checks=self._flann_checks)
            )
        if self._distance_type is MatchingDistanceType.EUCLIDEAN:
            distance_metric = cv.NORM_L2
        elif self._distance_type is MatchingDistanceType.HAMMING:
            distance_metric = cv.NORM_HAMMING
        else:
            raise NotImplementedError("The distance type is not in MatchingDistanceType")

        return cv.BFMatcher(normType=distance_metric, crossCheck=False)

    def __perform_matching(self, descriptors_1: np.ndarray, descriptors_2: np.ndarray) -> np.ndarray:
        """Run the core logic for matching.

        NOTE: cv2's BFMatcher parallelizes internally over the GLOBAL OpenCV thread pool, which
        defaults to all cores — with N dask workers that is N full-node thread pools thrashing
        each other (same oversubscription bug as pycolmap SIFT num_threads=-1, fixed 818f7b4f).
        Parallelism comes from Dask tasks or the caller's matching pool; each match stays single-threaded.

        Args:
            descriptors_1: descriptors for the 1st image.
            descriptors_2: descriptors for the 2nd image.

        Returns:
            indices of the match between two images.
        """
        cv.setNumThreads(1)  # idempotent; must run in the WORKER process (init doesn't re-run there)

        match_indices_1to2: Dict[int, int] = self.__perform_oneway_matching(descriptors_1, descriptors_2)
        match_indices_2to1: Dict[int, int] = self.__perform_oneway_matching(descriptors_2, descriptors_1)

        twoway_match_indices = np.array(
            [(idx1, idx2) for idx1, idx2 in match_indices_1to2.items() if match_indices_2to1.get(idx2) == idx1],
            dtype=np.uint32,
        )
        return twoway_match_indices

    def __perform_oneway_matching(self, descriptors_1: np.ndarray, descriptors_2: np.ndarray) -> Dict[int, int]:
        """Perform 1-way matching.

        Args:
            descriptors_1: descriptors for the 1st image.
            descriptors_2: descriptors for the 2nd image.

        Returns:
            indices of the match between two images as a dictionary.
        """
        opencv_matcher = self.__init_opencv_matcher()

        if self._ratio_test_threshold is not None:
            all_matches = opencv_matcher.knnMatch(descriptors_1, descriptors_2, k=2)
            # knnMatch(k=2) yields <2 candidates per query when the train side has <2 descriptors
            # (near-featureless internet images) — those queries cannot pass a ratio test; skip them.
            matches = [
                m[0] for m in all_matches
                if len(m) == 2 and m[0].distance <= self._ratio_test_threshold * m[1].distance
            ]
        else:
            matches = opencv_matcher.match(descriptors_1, descriptors_2)

        matches = sorted(matches, key=lambda match: match.distance)
        match_indices = {m.queryIdx: m.trainIdx for m in matches}

        return match_indices
