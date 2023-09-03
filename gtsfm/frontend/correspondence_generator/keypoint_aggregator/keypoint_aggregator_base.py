"""Base class for keypoint aggregators.

Authors: John Lambert
"""

import abc
from typing import Dict, List, Tuple

import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata


class KeypointAggregatorBase(GTSFMProcess):
    """Base class for keypoint aggregators."""

    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="KeypointAggregator",
            input_products=("Detections i", "Detections j"),
            output_products=("Keypoints", "Putative Correspondences"),
            parent_plate="ImageCorrespondenceGenerator",
        )

    @abc.abstractmethod
    def aggregate(
        self, keypoints_dict: Dict[Tuple[int, int], Tuple[Keypoints, Keypoints]]
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Aggregates per-pair image keypoints into a set of keypoints per image.

        Args:
            keypoints_dict: Dictionary where key (i1,i2) maps to (keypoints_i1, keypoints_i2) representing matches
                (correspondences).

        Returns:
            keypoints_list: List of N Keypoints objects for N images.
            putative_corr_idxs_dict: Mapping from image pair (i1,i2) to putative correspondence indices.
              Correspondence indices are represented by an array of shape (K,2), for K correspondences.
        """
