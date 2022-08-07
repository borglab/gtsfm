"""

Authors: John Lambert
"""

import abc
from typing import Dict, List, Optional, Tuple

import dask
from dask.delayed import Delayed

from gtsfm.common.keypoints import Keypoints


class KeypointAggregatorBase:
    @abc.abstractmethod
    def run(self, keypoints_dict: Dict[Tuple[int, int], Tuple[Keypoints, Keypoints]]) -> List[Optional[Keypoints]]:
        """
        Args:
            keypoints_dict: (i1,i2) maps to (keypoints_i1, keypoints_i2) representing matches (correspondences).

        Returns:
            keypoints_list: list of N Keypoints objects for N images.
            putative_corr_idxs_dict: putative correspondence indices (K,2) for each image pair (i1,i2).
        """

    def create_computation_graph(
        self, delayed_keypoints_dict: Dict[Tuple[int, int], Delayed]
    ) -> Tuple[List[Delayed], Dict[Tuple[int, int], Delayed]]:
        """Create Dask graph for keypoint aggregation from direct image feature matchers.

        Args:
            keypoints_dict: key (i1,i2) maps to (keypoints_i1, keypoints_i2) representing matches (correspondences).

        Returns:
            keypoints_list: list of N delayed tasks, each yielding a Keypoints object for one of the N images.
            putative_corr_idxs_dict: dictionary of delayed tasks. Each task evaluates to putative correspondence
                indices (K,2) for each image pair (i1,i2).
        """
        return dask.delayed(self.run, nout=2)(delayed_keypoints_dict)
