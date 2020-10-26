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
        for (pose_idx_1, pose_idx_2), feature_idxs in matches.items():
            for i in range(len(feature_idxs)):
                feature_idx_1 = feature_idxs[i][0]
                feature_idx_2 = feature_idxs[i][1]
                dsf.merge(gtsam.IndexPair(pose_idx_1, feature_idx_1),
                        gtsam.IndexPair(pose_idx_2, feature_idx_2))
                key_set = dsf.sets()                

        # create a landmark map
        for idx, s in enumerate(key_set):
            key = key_set[s]
            # Initialize track
            track = gtsam.SfmTrack()
            for val in gtsam.IndexPairSetAsArray(key):
                
                pose_idx = val.i()
                feature_idx = val.j()

                # get set representative- Will be IndexPair type
                lndmrk_root_node = dsf.find(gtsam.IndexPair(pose_idx, feature_idx))
                # for each representative, add (img_idx, feature)
                # add measurement in this track
                meas = tuple((pose_idx, feature_list[pose_idx][feature_idx][:2]))
                track.add_measurement(meas)

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
        for track_idx in range(len(landmark_data)):
            unique_track = set()
            for measurement_idx in range(landmark_data[track_idx].number_measurements()):
                i, _ = landmark_data[track_idx].measurement(measurement_idx)
                unique_track.add(i)
            if len(unique_track) != landmark_data[track_idx].number_measurements():
                continue
            else:
                filtered_landmark_data.append(landmark_data[track_idx])
        
        return filtered_landmark_data
