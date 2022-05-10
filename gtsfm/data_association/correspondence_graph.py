"""Implements the CorrespondenceGraph class.

Authors: Travis Driver
"""
from typing import Tuple, Dict, List
import itertools

import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.common.sfm_track import SfmTrack2d

import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()


class CorrespondenceGraph:
    """ """

    def __init__(
        self,
        corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
    ) -> None:
        """
        Args:
            corr_idxs_dict: dictionary, with key as image pair (i1,i2) and value as matching keypoint indices.
            keypoints_list: keypoints for each image.
        """
        self.num_images = len(corr_idxs_dict)
        self.tracks_2d = SfmTrack2d.generate_tracks_from_pairwise_matches(corr_idxs_dict, keypoints_list)

    def obs_from_tracks(self, frame_id: int) -> Dict[int, np.ndarray]:
        """Generates a dictionary of observations, i.e., track index and keypoint coordinate pairs, from frame i."""
        obs: Dict[int, np.ndarray] = {}
        for track_idx, track in enumerate(self.tracks_2d):
            for meas in track.measurements:
                if meas.i == frame_id:
                    obs[track_idx] = meas.uv
        return obs

    def get_aggregate_assoc_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        # Get list of items observed in each view.
        tmp: Dict[int, List[int]] = {i: [] for i in range(self.num_images)}
        for idx, track in enumerate(self.tracks_2d):
            for meas in track.measurements:
                tmp[meas.i].append(idx)

        # Build aggregate association matrix.
        items_per_view = [len(tmp[i]) for i in range(len(tmp))]
        items_sum_view = [int(np.sum(items_per_view[:i])) for i in range(len(tmp) + 1)]
        PP = np.eye(int(np.sum(items_sum_view[-1])))
        for idx, track in enumerate(self.tracks_2d):
            assoc_ind = []
            for meas in track.measurements:
                k = tmp[meas.i].index(idx)
                assoc_ind.append(items_sum_view[meas.i] + k)
            assoc_pairs = np.asarray(list(itertools.permutations(assoc_ind, 2)))
            for ij in assoc_pairs:
                PP[ij[0], ij[1]] = 1

        return PP, np.asarray(items_per_view)

    @staticmethod
    def from_assoc_matrix(PP: np.ndarray) -> "CorrespondenceGraph":
        """ """
