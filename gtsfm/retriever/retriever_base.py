"""Base class for the Retriever, which provides a list of potential image pairs.

Authors: John Lambert
"""

import abc
from enum import Enum
from typing import List, Tuple

import dask
from dask.delayed import Delayed

from gtsfm.loader.loader_base import LoaderBase


class ImageMatchingRegime(str, Enum):
    SEQUENTIAL: str = "sequential"
    RETRIEVAL: str = "retrieval"
    EXHAUSTIVE: str = "exhaustive"
    SEQUENTIAL_WITH_RETRIEVAL: str = "sequential_with_retrieval"


class RetrieverBase:
    def __init__(self, num_matched: int = 2) -> None:
        """
        Args:
            num_matched: number of K potential matches to provide per query. These are the top "K" matches per query.
        """
        self._num_matched = num_matched

    @abc.abstractmethod
    def run(loader: LoaderBase) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader. The length of this loader will provide the total number of images
                for exhaustive global descriptor matching.

        Return:
            pair_indices: (i1,i2) image pairs.
        """

    def create_computation_graph(self, loader: LoaderBase) -> Delayed:
        """Compute potential image pairs.

        Args:
            loader: image loader. The length of this loader will provide the total number of images
                for exhaustive global descriptor matching.

        Return:
            Delayed task that evaluates to a list of (i1,i2) image pairs.
        """
        return dask.delayed(self.run)(loader=loader)
