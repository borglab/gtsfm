"""Base class for correspondence generators.

Authors: John Lambert
"""

from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from dask.distributed import Client, Future

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.products.visibility_graph import VisibilityGraph


class CorrespondenceGeneratorBase:
    """Base class for correspondence generators."""

    @abstractmethod
    def generate_correspondences(
        self,
        client: Client,
        images: List[Future],
        visibility_graph: VisibilityGraph,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Apply the correspondence generator to generate putative correspondences.

        Args:
            client: Dask client, used to execute the front-end as futures.
            images: List of all images, as futures.
            visibility_graph: The visibility graph defining which image pairs to process.

        Returns:
            List of keypoints, one entry for each input images.
            Putative correspondence as indices of keypoints, for pairs of images.
        """

    @abstractmethod
    def generate_correspondences_inline(
        self,
        images: List[Image],
        visibility_graph: VisibilityGraph,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Generate putative correspondences inline, i.e. without submitting Dask tasks.

        Same contract as ``generate_correspondences`` but takes concrete images and runs detection/matching
        in plain loops in the calling process. This is the entry point for the per-cluster frontend, which
        already executes inside a Dask task: submitting further tasks from there (``worker_client()``)
        keeps every feature/correspondence future of the cluster resident on one worker during the nested
        gather.

        Args:
            images: Materialized images, indexed by position (``images[i]`` is image ``i``).
            visibility_graph: The visibility graph defining which image pairs to process.

        Returns:
            List of keypoints, one entry for each input images.
            Putative correspondence as indices of keypoints, for pairs of images.
        """
