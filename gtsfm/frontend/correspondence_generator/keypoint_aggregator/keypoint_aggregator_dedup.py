"""Keypoint aggregator that de-duplicates keypoint detections, aggregating from image pairs to single images.

Authors: John Lambert
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_base import KeypointAggregatorBase

logger = logger_utils.get_logger()


class KeypointAggregatorDedup(KeypointAggregatorBase):
    """Keypoint aggregator with de-duplication."""

    def run(self, keypoints_dict: Dict[Tuple[int, int], Tuple[Keypoints, Keypoints]]) -> List[Optional[Keypoints]]:
        """Aggregates per-pair image keypoints into a set of keypoints per image, with de-duplication.

        Keypoints are computed per image pair, instead of per image, so they are aggregated per image here.

        Args:
            keypoints_dict: (i1,i2) maps to (keypoints_i1, keypoints_i2) representing matches (correspondences).

        Returns:
            keypoints_list: list of N Keypoints objects for N images.
            putative_corr_idxs_dict: mapping from image pair (i1,i2) to putative correspondence indices.
              Correspondence indices are represented by an array of shape (K,2), for K correspondences.
        """
        image_indices = set()
        for (i1, i2) in keypoints_dict.keys():
            image_indices.add(i1)
            image_indices.add(i2)

        max_img_idx = max(image_indices)
        duplicates_found = 0

        putative_corr_idxs_dict = {}
        per_image_kpt_coordinates = {i: np.zeros((0, 2)) for i in image_indices}

        # TODO(johnwlambert): use efficient algo (DSF) to find tracks once C++ variant implemented,
        # instead of O(N^2) algo below.
        # Have to merge keypoints across different views here (or turn off transitivity check).

        for (i1, i2), (keypoints_i1, keypoints_i2) in keypoints_dict.items():
            # both keypoints_i1 and keypoints_i2 have the same shape
            N_to_add = keypoints_i1.coordinates.shape[0]

            N1 = per_image_kpt_coordinates[i1].shape[0]
            N2 = per_image_kpt_coordinates[i2].shape[0]

            i1_count = N1
            i1_indices = np.zeros(N_to_add)
            unique_keypoints_i1_coordinates = []

            for k1, uv1 in enumerate(keypoints_i1.coordinates):
                diff_norms = np.linalg.norm(per_image_kpt_coordinates[i1] - uv1, axis=1)
                is_identical1 = np.any(diff_norms == 0)
                if len(per_image_kpt_coordinates[i1]) > 0 and is_identical1:
                    duplicates_found += 1
                    i1_indices[k1] = np.argmin(diff_norms)
                else:
                    i1_indices[k1] = i1_count
                    i1_count += 1
                    unique_keypoints_i1_coordinates.append(uv1)

            i2_count = N2
            i2_indices = np.zeros(N_to_add)
            unique_keypoints_i2_coordinates = []

            for k2, uv2 in enumerate(keypoints_i2.coordinates):
                diff_norms = np.linalg.norm(per_image_kpt_coordinates[i2] - uv2, axis=1)
                is_identical2 = np.any(diff_norms == 0)
                if len(per_image_kpt_coordinates[i2]) > 0 and is_identical2:
                    duplicates_found += 1
                    i2_indices[k2] = np.argmin(diff_norms)
                else:
                    i2_indices[k2] = i2_count
                    i2_count += 1
                    unique_keypoints_i2_coordinates.append(uv2)

            unique_keypoints_i1_coordinates = np.array(unique_keypoints_i1_coordinates)
            unique_keypoints_i2_coordinates = np.array(unique_keypoints_i2_coordinates)

            if len(unique_keypoints_i1_coordinates) == 0:
                unique_keypoints_i1_coordinates = np.zeros((0, 2))

            if len(unique_keypoints_i2_coordinates) == 0:
                unique_keypoints_i2_coordinates = np.zeros((0, 2))

            per_image_kpt_coordinates[i1] = np.vstack([per_image_kpt_coordinates[i1], unique_keypoints_i1_coordinates])
            per_image_kpt_coordinates[i2] = np.vstack([per_image_kpt_coordinates[i2], unique_keypoints_i2_coordinates])

            putative_corr_idxs = np.stack([np.array(i1_indices), np.array(i2_indices)], axis=-1).astype(np.uint16)
            putative_corr_idxs_dict[(i1, i2)] = putative_corr_idxs

        logger.info(f"Merged {duplicates_found} duplicates during de-duplication.")

        keypoints_list = [None] * (max_img_idx + 1)
        for i in per_image_kpt_coordinates.keys():
            keypoints_list[i] = Keypoints(coordinates=per_image_kpt_coordinates[i])

        return keypoints_list, putative_corr_idxs_dict
