"""Base class for keypoint aggregators.

Authors: John Lambert
"""

import abc
from typing import Dict, List, Optional, Tuple

import dask
from dask.delayed import Delayed

from gtsfm.common.keypoints import Keypoints


class KeypointAggregatorBase:
    """Base class for keypoint aggregators."""

    @abc.abstractmethod
    def run(self, keypoints_dict: Dict[Tuple[int, int], Tuple[Keypoints, Keypoints]]) -> List[Optional[Keypoints]]:
        """Aggregates per-pair image keypoints into a set of keypoints per image.

        Args:
            keypoints_dict: (i1,i2) maps to (keypoints_i1, keypoints_i2) representing matches (correspondences).

        Returns:
            keypoints_list: list of N Keypoints objects for N images.
            putative_corr_idxs_dict: mapping from image pair (i1,i2) to putative correspondence indices.
              Correspondence indices are represented by an array of shape (K,2), for K correspondences.
        """

    def create_computation_graph(
        self, delayed_keypoints_dict: Dict[Tuple[int, int], Delayed]
    ) -> Tuple[List[Delayed], Dict[Tuple[int, int], Delayed]]:
        """Create Dask graph for keypoint aggregation from direct image feature matchers.

        Args:
            delayed_keypoints_dict: key (i1,i2) maps to (keypoints_i1, keypoints_i2) representing matches
                (correspondences).

        Returns:
            List of N delayed tasks, each yielding a Keypoints object for one of the N images.
            Dictionary of delayed tasks, each of which evaluates to putative correspondence
                indices (K,2) for each image pair (i1,i2).
        """
        return dask.delayed(self.run, nout=2)(delayed_keypoints_dict)
