"""Keypoint aggregator that assumes each detected keypoint per image pair will be unique across he entire image.

Authors: John Lambert
"""

from typing import Dict, List, Tuple

import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_base import KeypointAggregatorBase


class KeypointAggregatorUnique(KeypointAggregatorBase):
    """Keypoint aggregator without de-duplication, allowing for potentially duplicate keypoints per image."""

    def aggregate(
        self, keypoints_dict: Dict[Tuple[int, int], Tuple[Keypoints, Keypoints]]
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Aggregates per-pair image keypoints into a set of keypoints per image, without de-duplication.

        Assumes keypoint detections in each image pair are unique, and no de-duplication is necesary.

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

        # Determine length of the output list of Keypoints objects.
        max_img_idx = max(image_indices)

        putative_corr_idxs_dict = {}
        per_image_kpt_coordinates = {i: np.zeros((0, 2)) for i in image_indices}

        for (i1, i2), (keypoints_i1, keypoints_i2) in keypoints_dict.items():
            # both keypoints_i1 and keypoints_i2 have the same shape
            N_to_add = keypoints_i1.coordinates.shape[0]

            N1 = per_image_kpt_coordinates[i1].shape[0]
            N2 = per_image_kpt_coordinates[i2].shape[0]

            global_i1_indices = np.arange(N1, N1 + N_to_add)
            global_i2_indices = np.arange(N2, N2 + N_to_add)

            per_image_kpt_coordinates[i1] = np.vstack([per_image_kpt_coordinates[i1], keypoints_i1.coordinates])
            per_image_kpt_coordinates[i2] = np.vstack([per_image_kpt_coordinates[i2], keypoints_i2.coordinates])

            putative_corr_idxs = np.stack([np.array(global_i1_indices), np.array(global_i2_indices)], axis=-1).astype(
                np.uint16
            )
            putative_corr_idxs_dict[(i1, i2)] = putative_corr_idxs

        keypoints_list: List[Keypoints] = [Keypoints(coordinates=np.array([]))] * (max_img_idx + 1)
        for i in per_image_kpt_coordinates.keys():
            keypoints_list[i] = Keypoints(coordinates=per_image_kpt_coordinates[i])

        return keypoints_list, putative_corr_idxs_dict
