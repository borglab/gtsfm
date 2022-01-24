"""Image retriever for front-end.
Authors: Travis Driver, Jon Womack
"""
import abc
from typing import List, Tuple

import dask
from dask.delayed import Delayed

import gtsfm.utils.logger as logger_utils


logger = logger_utils.get_logger()


class RetrieverBase(metaclass=abc.ABCMeta):
    """Base class for image retrieval.
    The Retriever proposes image pairs to conduct local feature matching.
    """

    def __init__(self, image_pair_indices):
        """Initialize the Retriever

        :param image_pair_indices: Set of possible image pairs based on Dataloader
        """
        self.image_pair_indices = image_pair_indices

    @abc.abstractmethod
    def retrieve_potential_matches(self, image_graph: Delayed, num_closest_images) -> Delayed:
        """

        Args:
            image_graph:
            num_closest_images:
        Returns:
             List of potential matches in as List[(image_id, image_id)]
        """

    def create_computation_graph(self, image_graph: Delayed) -> List[Tuple[int, int]]:
        """Retrieve potential image matches

        Args:
            image_graph: List of GTSfM Image objects wrapped up in Delayed

        Returns:
            List of image index pairs
        """
        return dask.delayed(self.retrieve_potential_matches(image_graph))
