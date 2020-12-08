from data_association.feature_tracks import FeatureTrackGenerator
from data_association.data_assoc import DataAssociation
import gtsam
import numpy as np

from common.keypoints import Keypoints
from data_association.data_assoc import DataAssociation
from typing import List, Dict, Tuple

# randomly create tracks and assign arbitrary 3D landmark pts
# we only want to check if the inputs and outputs are as expected

class DummyDataAssociation(FeatureTrackGenerator):
    def __init__(self, 
                matches: Dict[Tuple[int, int], np.ndarray], 
                feature_list: List[Keypoints]):
        super().__init__(matches, feature_list)

    def run(self,         
            global_poses: List[gtsam.Pose3], 
            sharedcalibrationFlag: bool, 
            reprojection_threshold: float,
            min_track_length: int,
            use_ransac: bool,
            calibration: gtsam.Cal3Bundler, 
            camera_list: gtsam.CameraSetCal3Bundler,
            ) -> List:
        """ Inputs to Data Association module """

        triangulated_landmark_map = []        
        sfmdata_landmark_map = self.filtered_landmark_data
        
        # point indices are represented as j
        # nb of 3D points = nb of tracks, hence track_idx represented as j
        for j in range(len(sfmdata_landmark_map)):
            
            triangulated_data = {tuple((np.random.rand(), np.random.rand(), np.random.rand())): sfmdata_landmark_map[j]}
            filtered_track = gtsam.SfmTrack(list(triangulated_data.keys())[0])
            for triangulated_pt, track in triangulated_data.items():
                for (i, measurement) in track:
                    filtered_track.add_measurement(i, measurement)

            if filtered_track.number_measurements() >= min_track_length:
                triangulated_landmark_map.append(filtered_track)
            else:
                print("Track length < {} discarded".format(min_track_length))
                
        return triangulated_landmark_map
    

if __name__ == "__main__":
    