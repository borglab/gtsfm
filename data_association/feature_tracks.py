"""Class to hold the tracks (i.e. 2d measurements) of 3d landmarks.

Authors: Ayush Baid, Sushmita Warrier
"""
import numpy as np
import gtsam

from common.keypoints import Keypoints
from typing import Dict, List, Tuple

LANDMARK_MAP = List[List[Tuple[int,Tuple[float, float]]]]

class FeatureTrackGenerator:
    """
    Creates and filter tracks from matches.
    """

    def __init__(self,
                 matches: Dict[Tuple[int, int], np.ndarray],
                 feature_list: List[Keypoints]
                 ):
        """
        Creates DSF and landmark map from pairwise matches.

        Args:
            matches: Dict of pairwise matches of type:
                    key: pose indices for the matched pair of images
                    val: feature indices, as array of Nx2 shape; N being nb of features, and each row is (feature_idx1, feature_idx2).
            num_poses: Number of poses.
            feature_list: List of keypoints for each image.
        """

        # Generate the DSF to form tracks
        dsf = gtsam.DSFMapIndexPair()
        self.filtered_landmark_data = []
        landmark_data = []
        # for DSF finally
        # measurement_idxs represented by ks
        for (i1, i2), ks in matches.items():
            for idx in range(ks.shape[0]):
                k1 = ks[idx][0]
                k2 = ks[idx][1]
                dsf.merge(gtsam.IndexPair(i1, k1), gtsam.IndexPair(i2, k2))
                key_set = dsf.sets()                
        # create a landmark map: a list of tracks
        # Each track is represented as a list of (camera_idx, measurements)
        for s in key_set:
            key = key_set[s]
            # Initialize track
            track = []
            for val in gtsam.IndexPairSetAsArray(key):                
                # camera_idx is represented by i
                # measurement_idx is represented by k
                i = val.i()
                k = val.j()
                # add measurement in this track
                # check to ensure dimensions of coordinates are correct
                if feature_list[i].coordinates.ndim != 2:
                    raise Exception("Dimensions for Keypoint coordinates        incorrect. \
                                     Array needs to be 2D")
                track.append(tuple((i, feature_list[i].coordinates[k])))
            landmark_data.append(track)          
        self.filtered_landmark_data = self.delete_tracks(landmark_data)


    def delete_tracks(self, landmark_data: LANDMARK_MAP) -> LANDMARK_MAP:
        """
        Delete tracks that have more than one measurement in the same image.

        Args:
            landmark_data: List of feature tracks.
        Returns:
            list of filtered feature tracks. 
        """
        # TODO (Sush): Add inline comments to explain logic
        filtered_landmark_data = []
        # track_idx represented as j
        for j in range(len(landmark_data)):
            unique_pose_idxs = set()
            # measurement_idx represented as k
            for k in range(len(landmark_data[j])):
                i, _ = landmark_data[j][k]
                unique_pose_idxs.add(i)
            if len(unique_pose_idxs) != len(landmark_data[j]):
                continue
            else:
                filtered_landmark_data.append(landmark_data[j])
        
        return filtered_landmark_data
