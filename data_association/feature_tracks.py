"""Class to hold the tracks (i.e. 2d measurements) of 3d landmarks.

Authors: Ayush Baid, Sushmita Warrier
"""
from typing import OrderedDict, Dict, List, Tuple

import numpy as np
from collections import defaultdict

import gtsam
from numpy.core.defchararray import array
from numpy.core.records import ndarray


# class MeasurementToIndexMap:
#     def __init__(self):
#         self.__map = {}

#     def __generate_key(self,
#                        measurement: Tuple[float, float]) -> Tuple[int, int]:
#         # converts float position measurements to int
#         return tuple(map(lambda x: int(x), measurement))

#     def add(self, measurement: Tuple[float, float]):
#         key = self.__generate_key(measurement)

#         if key not in self.__map:
#             # If new feature, assign it new idx number
#             self.__map[key] = len(self.__map) # feature idx

#     def add_batch(self, measurements: np.ndarray):
#         for row in measurements:
#             self.add(tuple(row[:2])) #(x, y) positions
    

#     def lookup_index(self, measurement: Tuple[float, float]) -> int:
#         key = self.__generate_key(measurement)
#         # return value (ie. feature idx) corresponding to feature position
#         return self.__map[key]

#     def invert_map(self):
#         inv_map = {v: k for k, v in self.__map.items()}
#         return inv_map
    


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
        self.landmark_map = defaultdict(list)
        self.filtered_landmark_data = []
        landmark_data = []

        # for DSF finally
        for (pose_idx_1, pose_idx_2), feature_idxs in matches.items():
            print("len", len(feature_idxs))
            for i in range(len(feature_idxs)):
                feature_idx_1 = feature_idxs[i][0]
                print("idx check",feature_idxs[i], i)
                feature_idx_2 = feature_idxs[i][1]
                print("indices",pose_idx_1, pose_idx_2, feature_idx_1, feature_idx_2)
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
                print("pose", pose_idx, feature_idx)
                # get set representative- Will be IndexPair type
                lndmrk_root_node = dsf.find(gtsam.IndexPair(pose_idx, feature_idx))
                landmark_key = (lndmrk_root_node.i(), lndmrk_root_node.j())
                # for each representative, add (img_idx, feature)
                # feature is extracted from feature_idx by inverting dict mapping feature coordinates to idx

                # add measurement in this track
                print("check", feature_list[pose_idx][feature_idx][:2])
                meas = tuple((pose_idx, feature_list[pose_idx][feature_idx][:2]))
                track.add_measurement(meas)

            landmark_data.append(track)
            
        self.filtered_landmark_data = delete_tracks(landmark_data)


def delete_tracks(landmark_data: List) -> List:
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

def toy_case_2():
    matches = {(0,1): np.array([[0,2]]), 
                (1,2): np.array([[2,3], 
                                [4,5], 
                                [7,9]]),
                (0,2): np.array([[1,8]])}
    # feature_list format: [[features in img1], [features in img2], [features in img3]]
    feature_list = [
                    [(12,16, 6), (13,18, 9), (0,10, 8.5)], 
                    [(8,2), (16,14), (22,23), (1,6), (50,50), (16,12), (82,121), (39,60)], 
                    [(1,1), (8,13), (40,6), (82,21), (1,6), (12,18), (15,14), (25,28), (7,10), (14,17)]
                    ]
    return matches, feature_list

def toy_case():
    matches = {(0,1): [(0,2)], 
                (1,2): [(2,3), (4,5),(7,9)],
                (0,2): [(1,8)]}
    feature_list = [
                    [(12,16), (13,18), (0,10)], 
                    [(8,2), (16,14), (22,23), (1,6), (50,50), (16,12), (82,121), (39,60)], 
                    [(1,1), (8,13), (40,6), (82,21), (1,6), (12,18), (15,14), (25,28), (7,10), (14,17)]
                    ]
    return matches, feature_list

                
if __name__ == "__main__":
    dummy_matches, features = toy_case_2()
    # for a sanity check
    FT = FeatureTrackGenerator(dummy_matches, len(dummy_matches), features)
    print(FT.landmark_map)

    """
    pseudocode:
    1. populate sfmtracks with measurements of type(idx, imgPt)
    2. add all tracks in sfmdata
    3. view all tracks present in sfmdata
    4. view all measurements in sfmtracks
    5. Add a triangulated pt for particular track -> we need a setter function
    """