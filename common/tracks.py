"""Class to hold the tracks (i.e. 2d measurements) of 3d landmarks.

Authors: Ayush Baid
"""
from typing import Dict, List, Tuple

import numpy as np

import gtsam


class MeasurementToIndexMap:
    def __init__(self):
        self.__map = {}

    def __generate_key(self,
                       measurement: Tuple[float, float]) -> Tuple[int, int]:
        return tuple(map(lambda x: int(x), measurement))

    def add(self, measurement: Tuple[float, float]):
        key = self.__generate_key(measurement)

        if key not in self.__map:
            self.__map[key] = len(self.__map)

    def add_batch(self, measurements: np.ndarray):
        for row in measurements:
            self.add(tuple(row[:2]))

    def lookup_index(self, measurement: Tuple[float, float]) -> int:
        key = self.__generate_key(measurement)

        return self.__map[key]


class FeatureTracks:

    def __tracks_to_dsf(self, dsf: gtsam.DSFMapIndexPair) -> List:
        pass

    def __init__(self,
                 matches: Dict[Tuple[int, int], Tuple[np.ndarrayy, np.ndarray]],
                 num_poses: int,
                 global_poses: List[gtsam.Pose3]
                 ):
        self.tracks = []

        # Generate the DSF to form tracks
        dsf = gtsam.DSFMapIndexPair()

        measurement_to_index_maps = [MeasurementToIndexMap()] * num_poses

        # assign indices to measures for each camera pose
        for (pose_idx_1, pose_idx_2), (features_1, features_2) in matches.items():
            measurement_to_index_maps[pose_idx_1].add_batch(features_1)
            measurement_to_index_maps[pose_idx_2].add_batch(features_2)

        # for DSF finally
        for (pose_idx_1, pose_idx_2), (features_1, features_2) in matches.items():
            num_matches = features_1.shape[0]

            for match_idx in range(num_matches):
                feature_idx_1 = measurement_to_index_maps[pose_idx_1].lookup_index(
                    tuple(features_1[match_idx, :2]))

                feature_idx_2 = measurement_to_index_maps[pose_idx_2].lookup_index(
                    tuple(features_2[match_idx, :2]))

                dsf.merge(gtsam.IndexPair(pose_idx_1, feature_idx_1),
                          gtsam.IndexPair(pose_idx_2, feature_idx_2))
