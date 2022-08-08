"""Base class for correspondence generators.

Authors: John Lambert
"""

import abc
from typing import Dict, List, Tuple

from dask.delayed import Delayed
import numpy as np

from gtsfm.common.keypoints import Keypoints


class CorrespondenceGeneratorBase:
    """Base class for correspondence generators."""

    @abc.abstractmethod
    def create_computation_graph(
        self,
        image_graph: List[Delayed],
        image_shapes: List[Tuple[int, int]],
        image_pair_indices: List[Tuple[int, int]],
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Create Dask computation graph for correspondence generation.

        Args:
            image_graph: list of N images.
            image_shapes: list of N image shapes, as tuples (height,width) in pixels.
            image_pair_indices: list of image pairs, each represented by a tuple (i1,i2).

        Return:
            delayed_keypoints: list of keypoints, for each image.
            delayed_putative_corr_idxs_dict: mapping from image pair (i1,i2) to putative correspondence indices.
              Correspondence indices are represented by an array of shape (K,2), for K correspondences.
        """
