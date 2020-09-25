import abc
from typing import DefaultDict, Dict, List, Tuple, Bool

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

    def __init__(self, calibrationFlag: Bool, global_poses: List[gtsam.Pose3]) -> None:
        self.sharedCal_Flag = calibrationFlag
        self.__global_pose_list = global_poses
    
    def triangulate_landmark(self, calibration, img_measurements):
        """
        calibration: Can be sharedCal or array of individual camera calibrations
        img_measurements: List of observations (Point2)
        """
        # if shared calibration provided for all cameras
        if self.sharedCal_Flag:
            return gtsam.triangulatePoint3(self.__global_pose_list, calibration, img_measurements)
        return gtsam.triangulatePoint3()
    




