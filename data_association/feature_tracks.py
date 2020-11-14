"""Class to hold the tracks (i.e. 2d measurements) of 3d landmarks.

Authors: Ayush Baid, Sushmita Warrier
"""
from typing import OrderedDict, Dict, List, Tuple

import numpy as np
from collections import defaultdict

import gtsam

class FeatureTrackGenerator:
    """
    Creates and filter tracks from matches.
    """

    def __init__(self,
                 matches: Dict[Tuple[int, int], List[Tuple[int, int]]],
                 num_poses: int,
                 feature_list: List[List]
                 ):
        """
        Creates DSF and landmark map from pairwise matches.
        Args:
            matches: Dict of pairwise matches of type:
                    key: pose indices for the matched pair of images
                    val: feature indices for row-wise matched pairs
            num_poses: Number of poses
            feature_list: List of feature arrays for each pose
        """

        # Generate the DSF to form tracks
        dsf = gtsam.DSFMapIndexPair()
        self.filtered_landmark_data = []
        landmark_data = []

        # for DSF finally
        # measurement_idxs represented by k
        for (i1, i2), k in matches.items():
            for idx in range(len(k)):
                k1 = k[idx][0]
                k2 = k[idx][1]
                dsf.merge(gtsam.IndexPair(i1, k1),
                        gtsam.IndexPair(i2, k2))
                key_set = dsf.sets()                

        # create a landmark map
        for idx, s in enumerate(key_set):
            key = key_set[s]
            # Initialize track
            # track = gtsam.SfmTrack()
            track = []
            for val in gtsam.IndexPairSetAsArray(key):
                
                # camera_idx is represented by i
                # measurement_idx is represented by k
                i = val.i()
                k = val.j()

                # get set representative- Will be IndexPair type
                lndmrk_root_node = dsf.find(gtsam.IndexPair(i, k))
                # for each representative, add (img_idx, feature)
                # add measurement in this track
                track.append(tuple((i, feature_list[i][k][:2])))

            landmark_data.append(track)
            
        self.filtered_landmark_data = self.delete_tracks(landmark_data)


    def delete_tracks(self, landmark_data: List) -> List:
        """
        Delete tracks that have more than one measurement in the same image
        Args:
            landmark_data: List of SfmTrack
        Returns:
            list of filtered SfmTrack 
        """
        filtered_landmark_data = []
        # track_idx represented as j
        for j in range(len(landmark_data)):
            unique_track = set()
            # measurement_idx represented as k
            for k in range(len(landmark_data[j])):
                i, _ = landmark_data[j][k]
                unique_track.add(i)
            if len(unique_track) != len(landmark_data[j]):
                continue
            else:
                filtered_landmark_data.append(landmark_data[j])
        
        return filtered_landmark_data
