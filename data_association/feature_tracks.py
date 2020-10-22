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
                 matches: OrderedDict[Tuple[int, int], Tuple[int, int]],
                 num_poses: int,
                 feature_list: List[List[ndarray]]
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
        self.filtered_landmark_data = gtsam.SfmData()
        landmark_data = gtsam.SfmData()

        # for DSF finally
        for (pose_idx_1, pose_idx_2), (feature_idx_1, feature_idx_2) in matches.items():
        #     num_matches = features_1.shape[0]

        #     for match_idx in range(num_matches):
        #         feature_idx_1 = measurement_to_index_maps[pose_idx_1].lookup_index(
        #             tuple(features_1[match_idx, :2]))  #lookup indx input: (x,y) for all rows

        #         feature_idx_2 = measurement_to_index_maps[pose_idx_2].lookup_index(
        #             tuple(features_2[match_idx, :2]))


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
                landmark_key = (lndmrk_root_node.i(), lndmrk_root_node.j())
                # for each representative, add (img_idx, feature)
                # feature is extracted from feature_idx by inverting dict mapping feature coordinates to idx

                # add measurement in this track
                meas = tuple((pose_idx, feature_list[pose_idx][feature_idx, :2]))
                track.add_measurement(meas)

            landmark_data.add_track(track)
            
        self.filtered_landmark_data = delete_tracks(landmark_data)


def delete_tracks(landmark_data: gtsam.SfmData) -> gtsam.SfmData:
    filtered_landmark_data = gtsam.SfmData()
    for track_idx in range(landmark_data.number_tracks()):
        unique_track = set()
        for measurement_idx in range(landmark_data.track(track_idx).number_measurements()):
            i, _ = landmark_data.track(track_idx).measurement(measurement_idx)
            unique_track.add(i)
        if len(unique_track) != landmark_data.track(track_idx).number_measurements():
            continue
        else:
            filtered_landmark_data.add_track(landmark_data.track(track_idx))
    
    return filtered_landmark_data

def toy_case():
    matches = dict()
    key_list = [(0,1), (0,2), (1,2)]
    #feature_list = [[np.array([1,3])]]
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
    print(FT.landmark_map)

    """
    pseudocode:
    1. populate sfmtracks with measurements of type(idx, imgPt)
    2. add all tracks in sfmdata
    3. view all tracks present in sfmdata
    4. view all measurements in sfmtracks
    5. Add a triangulated pt for particular track -> we need a setter function
    """