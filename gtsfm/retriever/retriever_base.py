"""Base class for the Retriever, which provides a list of potential image pairs.

Authors: John Lambert
"""

import abc
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np

from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.products.visibility_graph import VisibilityGraph
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
            output_products=("Visibility Graph",),
            parent_plate="Loader and Retriever",
        )

    def set_max_frame_lookahead(self, n) -> None:
        """If supported, set the maximum frame lookahead for sequential matching."""
        raise AttributeError(f"{type(self).__name__} has no max_frame_lookahead")

    def set_num_matched(self, n) -> None:
        """If supported, set the maximum number of matched frames for similarity matching."""
        raise AttributeError(f"{type(self).__name__} has no num_matched")

    @abc.abstractmethod
    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> VisibilityGraph:
        """Compute potential image pairs.

        Args:
            global_descriptors: the global descriptors for the retriever, if needed.
            image_fnames: file names of the images
            plots_output_dir: Directory to save plots to. If None, plots are not saved.

        Returns:
            Visibility graph representing image pair connections.
        """

    def evaluate(self, num_images, visibility_graph: VisibilityGraph) -> GtsfmMetricsGroup:
        """Evaluates the retriever result.

        Args:
            num_images: the number of images in the dataset.
            visibility_graph: The visibility graph representing image pair connections.

        Returns:
            Retriever metrics group.
        """
        metric_group_name = "retriever_metrics"
        retriever_metrics = GtsfmMetricsGroup(
            metric_group_name,
            [
                GtsfmMetric("num_input_images", num_images),
                GtsfmMetric("num_retrieved_image_pairs", len(visibility_graph)),
            ],
        )
        return retriever_metrics
