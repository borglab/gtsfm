"""Base class for the Retriever, which provides a list of potential image pairs.

Authors: John Lambert
"""

import abc
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

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

    def __init__(self, matching_regime: ImageMatchingRegime) -> None:
        """
        Args:
            matching_regime: identifies type of matching used for image retrieval.
        """
        self._matching_regime = matching_regime

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="Image Retriever",
            input_products="Image Loader",
            output_products="Image Pair Indices",
            parent_plate="Loader and Retriever",
        )

    @abc.abstractmethod
    def get_image_pairs(self, loader: LoaderBase, plots_output_dir: Optional[Path] = None) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader. The length of this loader will provide the total number of images
                for exhaustive global descriptor matching.
            plots_output_dir: Directory to save plots to.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
