"""Default image retriever for front-end.
Authors: Travis Driver, Jon Womack
"""
from typing import List, Tuple

import dask
from dask.delayed import Delayed

from gtsfm.frontend.retriever.retriever_base import RetrieverBase
import gtsfm.utils.logger as logger_utils


logger = logger_utils.get_logger()


class SequentialRetriever(RetrieverBase):
    """Class for sequential image retrieval. Keeps all image pairs from Dataloader."""

    def __init__(self, image_pair_indices):
        """Initialize the Retriever.
        Args:
        """
        super().__init__(image_pair_indices)

    def retrieve_potential_matches(self) -> Delayed:
        """Retrieves all image pairs from Dataloader as potential matches.

        Args:
        Returns:
            image_pair_indices: List of retrieved image index pairs (same as from DataLoader)

        """
        return self.image_pair_indices

    def create_computation_graph(self) -> List[Tuple[int, int]]:
        """Retrieve potential image matches

        Args:

        Returns:
            List of image index pairs
        """
        return dask.delayed(self.retrieve_potential_matches())
