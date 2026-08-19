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

    # Whether generate_correspondences() returns ALREADY-VERIFIED correspondences (e.g. read from a COLMAP
    # database's two_view_geometries). When True, the verified pipeline reads them directly in the main
    # process and skips its Dask two-view estimation + the all-results gather (which OOMs at scale).
    # Default False: correspondences are putative and must be geometrically verified downstream.
    produces_verified_correspondences: bool = False

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

    def generate_correspondences_inline(
        self,
        images: List[Image],
        visibility_graph: VisibilityGraph,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Inline (no-Dask) correspondence generation over already-materialized images.

        Same contract as ``generate_correspondences`` but takes concrete ``Image`` objects (indexed by
        position) and runs detection/matching in plain loops — no ``client.submit``/``gather``. This is the
        cluster-frontend path; it avoids the ``worker_client()`` tasks-launching-tasks pattern that holds
        every feature/correspondence future resident on one worker during a nested gather.

        Subclasses used by the cluster pipeline must override this. The default raises so an unsupported
        generator fails loudly rather than silently falling back to the fragile path.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_correspondences_inline()."
        )
