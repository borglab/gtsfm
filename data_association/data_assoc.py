import abc
from typing import DefaultDict, Dict, List, Tuple, Bool, Optional

import dask
import numpy as np
import cv2

import gtsam

class DataAssociation(metaclass=abc.ABCMeta):
    pass

class LandmarkInitialization(metaclass=abc.ABCMeta):
    """
    Class to initialize landmark points via triangulation
    """

    def __init__(self, calibrationFlag: Bool, track_poses: List[gtsam.Pose3] = None, track_cameras: List[gtsam.Cal3_S2] = None) -> None:
        self.sharedCal_Flag = calibrationFlag
        self.__pose_list = track_poses
        # for shared calibration
        if track_poses is not None:
            self.track_pose_list = track_poses
        # for multiple cameras with individual calibrations
        if track_cameras is not None:
            self.track_camera_list = track_cameras
    
    

    def triangulate_landmark(self, calibration, img_measurements) -> gtsam.Point3:
        """
        Args:
            calibration: Can be sharedCal or array of individual camera calibrations
            img_measurements: List of observations (Point2)
        Returns:
            triangulated_landmark: triangulated landmark
        """
        # if shared calibration provided for all cameras
        if self.sharedCal_Flag:
            return gtsam.triangulatePoint3(self.__pose_list, calibration, img_measurements)
        else:
            return gtsam.triangulatePoint3(self.track_camera_list, img_measurements)
    


