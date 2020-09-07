"""Class to hold the tracks (i.e. 2d measurements) of 3d landmarks.

Authors: Ayush Baid
"""
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
from collections import defaultdict

import gtsam
from numpy.core.defchararray import array
from numpy.core.records import ndarray


class MeasurementToIndexMap:
    def __init__(self):
        self.__map = {}

    def __generate_key(self,
                       measurement: Tuple[float, float]) -> Tuple[int, int]:
        # converts float position measurements to int
        return tuple(map(lambda x: int(x), measurement))

    def add(self, measurement: Tuple[float, float]):
        key = self.__generate_key(measurement)

        if key not in self.__map:
            # If new feature, assign it new idx number
            self.__map[key] = len(self.__map) # feature idx

    def add_batch(self, measurements: np.ndarray):
        for row in measurements:
            self.add(tuple(row[:2])) #(x, y) positions
    

    def lookup_index(self, measurement: Tuple[float, float]) -> int:
        key = self.__generate_key(measurement)
        # return value (ie. feature idx) corresponding to feature position
        return self.__map[key]

    def invert_map(self):
        inv_map = {v: k for k, v in self.__map.items()}
        return inv_map
    


class FeatureTracks:

    def __tracks_to_dsf(self, dsf: gtsam.DSFMapIndexPair) -> List:
        pass


    def __init__(self,
                 matches: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
                 num_poses: int,
                 global_poses: List[gtsam.Pose3]
                 ):

        # Generate the DSF to form tracks
        dsf = gtsam.DSFMapIndexPair()
        key_set = set()
        self.landmark_map = defaultdict(list)

        measurement_to_index_maps = [MeasurementToIndexMap()] * num_poses

        # assign indices to measures for each camera pose
        # feature coordinates converted to indices for gtsamIndexPair input
        for (pose_idx_1, pose_idx_2), (features_1, features_2) in matches.items():
            measurement_to_index_maps[pose_idx_1].add_batch(features_1)
            measurement_to_index_maps[pose_idx_2].add_batch(features_2)

        # for DSF finally
        for (pose_idx_1, pose_idx_2), (features_1, features_2) in matches.items():
            num_matches = features_1.shape[0]

            for match_idx in range(num_matches):
                feature_idx_1 = measurement_to_index_maps[pose_idx_1].lookup_index(
                    tuple(features_1[match_idx, :2]))  #lookup indx input: (x,y) for all rows

                feature_idx_2 = measurement_to_index_maps[pose_idx_2].lookup_index(
                    tuple(features_2[match_idx, :2]))


                dsf.merge(gtsam.IndexPair(pose_idx_1, feature_idx_1),
                          gtsam.IndexPair(pose_idx_2, feature_idx_2))
                # Need to expose gtsam set to create key_set
                key_set.add((pose_idx_1, feature_idx_1))
                key_set.add((pose_idx_2, feature_idx_2))
                

        # create a landmark map
        for key in key_set:
            pose_idx, feature_idx = key
            # get set representative- Will be IndexPair type
            lndmrk_root_node = dsf.find(gtsam.IndexPair(pose_idx, feature_idx))
            landmark_key = (lndmrk_root_node.i(), lndmrk_root_node.j())
            # for each representative, add (img_idx, feature)
            # feature is extracted from feature_idx by inverting dict mapping feature coordinates to idx
            feature_dict = measurement_to_index_maps[pose_idx].invert_map()
            self.landmark_map[landmark_key].append((pose_idx, feature_dict[feature_idx]))

def toy_case():
    matches = dict()
    key_list = [(0,1), (0,2), (1,2)]
    matches_list = [
        (
            np.array([[1,3,2], [4,6,2], [9,8,1]]),
            np.array([[8,2,2], [5,10,2], [11,12,2]])
        ), 
        (
            np.array([[4,6,2]]),
            np.array([[12,14,1]])
        ),
        (
            np.array([[8,2,2], [4,1,5]]), 
            np.array([[13,16,2],[8,1,3]])
        )]
    for i,key in enumerate(key_list):
        matches[key] = matches_list[i]
    return matches

                
if __name__ == "__main__":
    dummy_matches = toy_case()
    # for a sanity check
    FT = FeatureTracks(dummy_matches, len(dummy_matches), None)