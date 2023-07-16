"""Keypoint aggregator that de-duplicates keypoint detections, aggregating from image pairs to single images.

Authors: John Lambert
"""

from typing import Dict, List, Tuple

import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_base import KeypointAggregatorBase

logger = logger_utils.get_logger()

KEYPOINT_DEDUP_THRESHOLD_PX = 0.5


class KeypointAggregatorDedup(KeypointAggregatorBase):
    """Keypoint aggregator with de-duplication."""

    def __init__(self) -> None:
        """Initialize global variables"""
        self.duplicates_found = 0

    def append_unique_keypoints(
        self, i: int, keypoints: Keypoints, per_image_kpt_coordinates: Dict[Tuple[int, int], np.ndarray]
    ) -> Tuple[Dict[Tuple[int, int], np.ndarray], np.ndarray]:
        """Identify unique keypoints, and append them to running list of global keypoints per image.

        If duplicate keypoints are found, the index of the previously existing keypoint is recorded.

        Args:
           i: image frame index.
           keypoints: keypoints detected in single image, from direct image pair feature matching.
           per_image_kpt_coordinates: running list of global keypoints, per image.

        Returns:
            per_image_kpt_coordinates: running list of global keypoints, per image.
            i_indices: single column of putative correspondence indices, for pair. These represent indices
                into the global table of keypoints per image.
        """
        N_to_add = keypoints.coordinates.shape[0]
        i_indices = np.zeros(N_to_add)
        i_count = per_image_kpt_coordinates[i].shape[0]
        unique_keypoints_i_coordinates = []

        for k, uv in enumerate(keypoints.coordinates):
            diff_norms = np.linalg.norm(per_image_kpt_coordinates[i] - uv, axis=1)
            # TODO(johnwlambert,ayushbaid): test loosening threshold below to some epsilon.
            is_identical = np.any(diff_norms < KEYPOINT_DEDUP_THRESHOLD_PX)
            if len(per_image_kpt_coordinates[i]) > 0 and is_identical:
                self.duplicates_found += 1
                i_indices[k] = np.argmin(diff_norms)
            else:
                i_indices[k] = i_count
                i_count += 1
                unique_keypoints_i_coordinates.append(uv)

        unique_keypoints_i_coordinates = np.array(unique_keypoints_i_coordinates)

        if len(unique_keypoints_i_coordinates) == 0:
            unique_keypoints_i_coordinates = np.zeros((0, 2))

        per_image_kpt_coordinates[i] = np.vstack([per_image_kpt_coordinates[i], unique_keypoints_i_coordinates])
        return per_image_kpt_coordinates, np.array(i_indices)

    def aggregate(
        self, keypoints_dict: Dict[Tuple[int, int], Tuple[Keypoints, Keypoints]]
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Aggregates per-pair image keypoints into a set of keypoints per image, with de-duplication.

        Keypoints are computed per image pair, instead of per image, so they are aggregated per image here.

        Args:
            keypoints_dict: key (i1,i2) maps to (keypoints_i1, keypoints_i2) representing matches (correspondences).

        Returns:
            keypoints_list: list of N Keypoints objects for N images.
            putative_corr_idxs_dict: mapping from image pair (i1,i2) to putative correspondence indices.
              Correspondence indices are represented by an array of shape (K,2), for K correspondences.
        """
        image_indices = set()
        for i1, i2 in keypoints_dict.keys():
            image_indices.add(i1)
            image_indices.add(i2)

        max_img_idx = max(image_indices)

        putative_corr_idxs_dict = {}
        per_image_kpt_coordinates = {i: np.zeros((0, 2)) for i in image_indices}

        # TODO(johnwlambert): use efficient algo (DSF) to find tracks once C++ variant implemented,
        # instead of O(N^2) algo below.
        # Have to merge keypoints across different views here (or turn off transitivity check).

        for (i1, i2), (keypoints_i1, keypoints_i2) in keypoints_dict.items():
            per_image_kpt_coordinates, i1_indices = self.append_unique_keypoints(
                i=i1, keypoints=keypoints_i1, per_image_kpt_coordinates=per_image_kpt_coordinates
            )
            per_image_kpt_coordinates, i2_indices = self.append_unique_keypoints(
                i=i2, keypoints=keypoints_i2, per_image_kpt_coordinates=per_image_kpt_coordinates
            )
            putative_corr_idxs = np.stack([i1_indices, i2_indices], axis=-1).astype(np.uint16)
            putative_corr_idxs_dict[(i1, i2)] = putative_corr_idxs

        logger.info(f"Merged {self.duplicates_found} duplicates during de-duplication.")
        # Reset global state.
        self.duplicates_found = 0

        keypoints_list: List[Keypoints] = [Keypoints(coordinates=np.array([]))] * (max_img_idx + 1)
        for i in per_image_kpt_coordinates.keys():
            keypoints_list[i] = Keypoints(coordinates=per_image_kpt_coordinates[i])

        return keypoints_list, putative_corr_idxs_dict
