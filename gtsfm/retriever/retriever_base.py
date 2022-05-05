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
    EXHAUSTIVE: str = "exhaustive"


class RetrieverBase:
    """Base class for image retriever implementations."""

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
        """Create Dask graph for image retriever.

        Args:
            loader: image loader. The length of this loader will provide the total number of images
                for exhaustive global descriptor matching.

        Return:
            Delayed task that evaluates to a list of (i1,i2) image pairs.
        """
        return dask.delayed(self.run)(loader=loader)
