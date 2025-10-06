"""Base class for the Retriever, which provides a list of potential image pairs.

Authors: John Lambert
"""

import abc
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np

from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.products.visibility_graph import ImageIndexPairs
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata


class ImageMatchingRegime(str, Enum):
    SEQUENTIAL = "sequential"
    RETRIEVAL = "retrieval"
    EXHAUSTIVE = "exhaustive"
    SEQUENTIAL_WITH_RETRIEVAL = "sequential_with_retrieval"
    RIG_HILTI = "rig_hilti"
    SEQUENTIAL_HILTI = "sequential_hilti"


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
            input_products=("Image Loader",),
            output_products=("Image Pair Indices",),
            parent_plate="Loader and Retriever",
        )

    @abc.abstractmethod
    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> ImageIndexPairs:
        """Compute potential image pairs.

        Args:
            global_descriptors: the global descriptors for the retriever, if needed.
            image_fnames: file names of the images
            plots_output_dir: Directory to save plots to. If None, plots are not saved.

        Returns:
            List of (i1,i2) image pairs.
        """

    def evaluate(self, num_images, image_pair_indices: ImageIndexPairs) -> GtsfmMetricsGroup:
        """Evaluates the retriever result.

        Args:
            num_images: the number of images in the dataset.
            image_pair_indices: (i1,i2) image pairs.

        Returns:
            Retriever metrics group.
        """
        metric_group_name = "retriever_metrics"
        retriever_metrics = GtsfmMetricsGroup(
            metric_group_name,
            [
                GtsfmMetric("num_input_images", num_images),
                GtsfmMetric("num_retrieved_image_pairs", len(image_pair_indices)),
            ],
        )
        return retriever_metrics
