"""Base class for correspondence generators.

Authors: John Lambert
"""

import abc
from typing import Dict, List, Tuple

from dask.delayed import Delayed


class CorrespondenceGeneratorBase:
    """Base class for correspondence generators."""

    @abc.abstractmethod
    def create_computation_graph(
        self,
        delayed_images: List[Delayed],
        delayed_image_shapes: List[Delayed],
        image_pair_indices: List[Tuple[int, int]],
    ) -> Tuple[List[Delayed], Dict[Tuple[int, int], Delayed]]:
        """Create Dask computation graph for correspondence generation.

        Args:
            delayed_images: list of N images.
            delayed_image_shapes: list of N image shapes tuples (H,W) in pixels (Tuple[int, int] wrapped as Delayed).
            image_pair_indices: list of image pairs, each represented by a tuple (i1,i2).

        Return:
            delayed_keypoints: list of delayed tasks, each yielding Keypoints in one image.
            delayed_putative_corr_idxs_dict: mapping from image pair (i1,i2) to delayed task to compute
                putative correspondence indices. Correspondence indices are represented by an array of
                shape (K,2), for K correspondences.
        """
