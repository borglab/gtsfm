
import gtsam
import numpy as np
import dask

from common.keypoints import Keypoints
from typing import List, Dict, Tuple

# create tracks and assign predefined 3D landmark pts
# we only want to check if the inputs and outputs are as expected

class DummyDataAssociation():
    def __init__(self, 
                matches: Dict[Tuple[int, int], np.ndarray], 
                feature_list: List[Keypoints]):
        self.filtered_landmark_data = []
        for cam_idx, measurement_idx in matches.items():
            # dummy camera and measurement indices
            for m in range(2):
                k = measurement_idx[0][m]   
            # random track   
                self.filtered_landmark_data.append(tuple((cam_idx[m], feature_list[cam_idx[m]].coordinates[k])))

    def run(self,         
            global_poses: List[gtsam.Pose3], 
            sharedcalibrationFlag: bool, 
            reprojection_threshold: float,
            min_track_length: int,
            use_ransac: bool,
            calibration: gtsam.Cal3Bundler, 
            camera_list: gtsam.CameraSetCal3Bundler,
            ) -> List[gtsam.SfmTrack]:
        """ Inputs to Data Association module """

        triangulated_landmark_map = []        
        sfmdata_landmark_map = self.filtered_landmark_data
        # triangulated pts predefined here, to avoid randomness
        triangulated_pts = [tuple((5.002913072201889, 0.5012185579790354, 1.2007032270128029))]
        # point indices are represented as j
        # nb of 3D points = nb of tracks, hence track_idx represented as j
        #for j in range(len(sfmdata_landmark_map)):
            
        triangulated_data = {triangulated_pts[0]: sfmdata_landmark_map}
        filtered_track = gtsam.SfmTrack(list(triangulated_data.keys())[0])
        for triangulated_pt, track in triangulated_data.items():
            for (i, measurement) in track:
                filtered_track.add_measurement(i, measurement)
        # reprojection error filtering skipped
        if filtered_track.number_measurements() >= min_track_length:
            triangulated_landmark_map.append(filtered_track)
        else:
            print("Track length < {} discarded".format(min_track_length))
                
        return triangulated_landmark_map


    def create_computation_graph(self, 
        global_poses: List[gtsam.Pose3], 
        sharedcalibrationFlag: bool, 
        reprojection_threshold: float,
        min_track_length: int,
        use_ransac: bool,
        calibration: gtsam.Cal3Bundler, 
        camera_list: gtsam.CameraSetCal3Bundler,
        ):
        """ 
        Generates computation graph for data association 

        Args:
            global poses: list of poses.
            sharedcalibrationFlag: flag to set shared or individual calibration
            reprojection_threshold: error threshold for track filtering.
            min_track_length: Minimum nb of views that must support a landmark for it to be accepted.
            use_ransac: Select between simple triangulation(False) and ransac-based triangulation(True).
            calibration: shared calibration.
            camera_list: list of individual cameras (if calibration not shared).
        
        Returns:
            Delayed dask tasks for data association.
        """
        return dask.delayed(self.run)(global_poses, 
                                      sharedcalibrationFlag, reprojection_threshold, min_track_length, 
                                      use_ransac, 
                                      calibration, 
                                      camera_list) 
    
