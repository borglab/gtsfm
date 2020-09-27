import abc
from collections import defaultdict
from densify.metrics import landmark_dict
from typing import DefaultDict, Dict, List, Tuple, Optional

import dask
import numpy as np
import cv2

import gtsam
from data_association.tracks import FeatureTracks

class DataAssociation(FeatureTracks):
    """
    workflow: form feature tracks; for each track, call LandmarkInitialization
    """
    def __init__(self, matches, num_poses, global_poses, calibrationFlag, calibration, camera_list) -> None:
        filtered_map = super().__init__(matches, num_poses, global_poses)
        self.calibrationFlag = calibrationFlag
        self.calibration = calibration

        for landmark_key, feature_track in filtered_map.items():
            if self.calibrationFlag == True:
                LMI = LandmarkInitialization(calibrationFlag, feature_track, calibration,global_poses)
            else:
                LMI = LandmarkInitialization(calibrationFlag, feature_track, camera_list)
            triangulated_landmark = LMI.triangulate_landmark(feature_track)
            # Replace landmark_key with triangulated landmark
        

class LandmarkInitialization(metaclass=abc.ABCMeta):
    """
    Class to initialize landmark points via triangulation
    """

    def __init__(self, 
    calibrationFlag: bool,
    obs_list: List,
    calibration: Optional[gtsam.Cal3_S2] = None, 
    track_poses: Optional[List[gtsam.Pose3]] = None, 
    track_cameras: Optional[List[gtsam.Cal3_S2]] = None) -> None:
        """
        Args:
            calibrationFlag: check if shared calibration exists(True) or each camera has individual calibration(False)
            obs_list: Feature track of type [(img_idx, img_measurement),..]
            calibration: Shared calibration
            track_poses: List of poses in a feature track
            track_cameras: List of cameras in a feature track
        """
        self.sharedCal_Flag = calibrationFlag
        self.observation_list = obs_list
        # for shared calibration
        if track_poses is not None:
            self.track_pose_list = track_poses
        # for multiple cameras with individual calibrations
        if track_cameras is not None:
            self.track_camera_list = track_cameras
    
    
    def extract_end_poses(self, track) -> Tuple[List, List]:
        pose_estimates = []
        cameras_list = []
        for (img_idx, img_Pt) in track:
            if self.sharedCal_Flag:
                pose_estimates.append(self.track_pose_list[img_idx])
            else:
                cameras_list.append(self.track_camera_list[img_idx])    
        return pose_estimates, cameras_list


    def triangulate_landmark(self, track) -> gtsam.Point3:
        """
        Args:
            track: List of (img_idx, observations(Point2))
        Returns:
            triangulated_landmark: triangulated landmark
        """
        pose_estimates, camera_values = self.extract_end_poses(track)
        for (img_idx, img_measurements) in track:
            # if shared calibration provided for all cameras
            if self.sharedCal_Flag:
                if self.track_pose_list == None or not pose_estimates:
                    raise Exception('track_poses arg or pose estimates missing')
                return gtsam.triangulatePoint3(pose_estimates, self.calibration, img_measurements)
            else:
                if self.track_camera_list == None or not camera_values:
                    raise Exception('track_cameras arg or camera values missing')
                return gtsam.triangulatePoint3(camera_values, img_measurements)
    
