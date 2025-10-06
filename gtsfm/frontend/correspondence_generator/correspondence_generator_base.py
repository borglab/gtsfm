"""Base class for correspondence generators.

Authors: John Lambert
"""

from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from dask.distributed import Client, Future

from gtsfm.common.keypoints import Keypoints
from gtsfm.products.visibility_graph import ImageIndexPairs


class CorrespondenceGeneratorBase:
    """Base class for correspondence generators."""

    @abstractmethod
    def generate_correspondences(
        self,
        client: Client,
        images: List[Future],
        image_pairs: ImageIndexPairs,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Apply the correspondence generator to generate putative correspondences.

        Args:
            client: Dask client, used to execute the front-end as futures.
            images: List of all images, as futures.
            image_pairs: Indices of the pairs of images to estimate two-view pose and correspondences.

        Returns:
            List of keypoints, one entry for each input images.
            Putative correspondence as indices of keypoints, for pairs of images.
        """
