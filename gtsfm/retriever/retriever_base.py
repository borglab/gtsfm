"""Base class for the Retriever, which provides a list of potential image pairs.

Authors: John Lambert
"""

import abc
from enum import Enum
from typing import List, Tuple

import dask
from dask.delayed import Delayed

from gtsfm.loader.loader_base import LoaderBase

from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata


class ImageMatchingRegime(str, Enum):
    SEQUENTIAL: str = "sequential"
    RETRIEVAL: str = "retrieval"
    EXHAUSTIVE: str = "exhaustive"
    SEQUENTIAL_WITH_RETRIEVAL: str = "sequential_with_retrieval"
    RIG_HILTI: str = "rig_hilti"
    SEQUENTIAL_HILTI: str = "sequential_hilti"


class RetrieverBase(GTSFMProcess):
    """Base class for image retriever implementations."""

    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display this process in the process graph."""

        return UiMetadata("Image Retriever", "", ("Image Loader"), ("Image Pair Indices"))

    @abc.abstractmethod
    def run(self, loader: LoaderBase) -> List[Tuple[int, int]]:
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
