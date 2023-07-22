"""Utilility functions for correspondences.

Authors: Ayush Baid
"""
from typing import List, Tuple, Dict

import numpy as np

from gtsfm.common.keypoints import Keypoints


def merge_keypoints(keypoints_1: Keypoints, keypoints_2: Keypoints) -> Tuple[Keypoints, int]:
    if len(keypoints_1) == 0:
        return keypoints_2, 0
    if len(keypoints_2) == 0:
        return keypoints_1, len(keypoints_1)

    # Note(Ayush): we avoid augmenting scales and responses and they do not make sense across different detectors
    return (
        Keypoints(coordinates=np.append(keypoints_1.coordinates, keypoints_2.coordinates, axis=0)),
        len(keypoints_1),
    )


def merge_correspondences(
    keypoints: List[List[Keypoints]], corr_idxs: List[Dict[Tuple[int, int], np.ndarray]]
) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
    num_frontends = len(keypoints)
    if num_frontends == 0:
        return [], {}
    elif num_frontends == 1:
        return keypoints[0], corr_idxs[0]

    num_images = len(keypoints[0])

    # Init merged keypoints and corr idxs with the 1st entry.
    merged_keypoints: List[Keypoints] = keypoints[0]
    merged_corr_idxs: Dict[Tuple[int, int], np.ndarray] = corr_idxs[0]

    for frontend_idx in range(1, num_frontends):
        # Merge just the keypoints first.
        keypoints_idx_offset: List[int] = [0] * num_images
        for i in range(num_images):
            merged_keypoints[i], keypoints_idx_offset[i] = merge_keypoints(
                merged_keypoints[i], keypoints[frontend_idx][i]
            )

        # With the offset for the keypoints, merge the correspondences
        curr_corr_idxs = corr_idxs[frontend_idx]
        for (i1, i2), corr_idxs_2 in curr_corr_idxs.items():
            if len(corr_idxs_2) > 0:
                corr_idxs_2[:, 0] += keypoints_idx_offset[i1]
                corr_idxs_2[:, 1] += keypoints_idx_offset[i2]

                if (i1, i2) in merged_corr_idxs:
                    merged_corr_idxs[(i1, i2)] = np.append(merged_corr_idxs[(i1, i2)], corr_idxs_2, axis=0)
                else:
                    merged_corr_idxs[(i1, i2)] = corr_idxs_2

    return merged_keypoints, merged_corr_idxs
