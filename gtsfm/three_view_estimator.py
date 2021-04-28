"""Estimator which operates on a triplet of images to compute relative pose and verified indices.

Authors: Ayush Baid, John Lambert
"""
import logging

import dask
import numpy as np
from dask.delayed import Delayed

import gtsfm.utils.logger as logger_utils
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.verifier.verifier_base import VerifierBase

logger = logger_utils.get_logger()

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)


class ThreeViewEstimator:
    """Wrapper for running three-view relative pose estimation on image triplets in the dataset."""

    def __init__(self, matcher: MatcherBase, verifier: VerifierBase, corr_metric_dist_threshold: float) -> None:
        """Initializes the three-view estimator from matcher and verifier.

        Args:
            matcher: matcher to use for image pairs.
            verifier: verifier to use.
            corr_metric_dist_threshold: distance threshold for marking a correspondence pair as inlier. 
        """
        self._matcher = matcher
        self._verifier = verifier
        self._corr_metric_dist_threshold = corr_metric_dist_threshold

    @classmethod
    def match_triplet_using_pairwise_matches(
        cls, match_indices_i1i2: np.ndarray, match_indices_i2i3: np.ndarray, match_indices_i1i3: np.ndarray
    ) -> np.ndarray:
        """Form correspondences across triplets by using pairwise matches.

        Note: this logic may not apply to all types of matchers (e.g. ones with ratio test).

        Args:
            match_indices_i1i2: indices of matches between image i1 and i2, of shape (N1, 2).
            match_indices_i2i3: indices of matches between image i2 and i3, of shape (N2, 2).
            match_indices_i1i3: indices of matches between image i1 and i3, of shape (N3, 2).

        Returns:
            indices of matches between the triplet (i1, i2, i3), of shape (N, 3), where N <= min(N1, N2, N3).
        """
        i1_to_i2_dict = {}
        # TODO: remove this dictionary, not needed between i1 and i2
        for idx_i1, idx_i2 in match_indices_i1i2:
            i1_to_i2_dict[idx_i1] = idx_i2

        i2_to_i3_dict = {}
        for idx_i2, idx_i3 in match_indices_i2i3:
            i2_to_i3_dict[idx_i2] = idx_i3

        i1_to_i3_dict = {}
        for idx_i1, idx_i3 in match_indices_i1i3:
            i1_to_i3_dict[idx_i1] = idx_i3

        # for the triplet matches, chaining i1->i2 with i2->i3 should be present in i1->i3
        triplet_matches = []
        for idx_i1, idx_i2 in i1_to_i2_dict.items():
            if idx_i2 in i2_to_i3_dict:
                idx_i3_chained = i2_to_i3_dict[idx_i2]
                if i1_to_i3_dict.get(idx_i1, None) == idx_i3_chained:
                    triplet_matches.append((idx_i1, idx_i2, idx_i3_chained))

        return np.array(triplet_matches, dtype=np.int64)

    def get_corr_metric_dist_threshold(self) -> float:
        """Getter for the distance threshold used in the metric for correct correspondences."""
        return self._corr_metric_dist_threshold

    def create_computation_graph_for_pairwise_matches(
        self,
        keypoints_i1_graph: Delayed,
        keypoints_i2_graph: Delayed,
        descriptors_i1_graph: Delayed,
        descriptors_i2_graph: Delayed,
    ) -> Delayed:
        """Create delayed tasks for matching and verification.

        Args:
            keypoints_i1_graph: keypoints for image i1.
            keypoints_i2_graph: keypoints for image i2.
            descriptors_i1_graph: corr. descriptors for image i1.
            descriptors_i2_graph: corr. descriptors for image i2.

        Returns:
            Correspondence indices between i1 and i2, wrapped up in Delayed.
        """
        return self._matcher.create_computation_graph(
            keypoints_i1_graph, keypoints_i2_graph, descriptors_i1_graph, descriptors_i2_graph
        )

    def create_computation_graph_for_triplet_matches(
        self, match_indices_i1i2_graph: Delayed, match_indices_i2i3_graph: Delayed, match_indices_i1i3_graph: Delayed
    ) -> Delayed:

        return dask.delayed(self.match_triplet_using_pairwise_matches)(
            match_indices_i1i2_graph, match_indices_i2i3_graph, match_indices_i1i3_graph
        )
