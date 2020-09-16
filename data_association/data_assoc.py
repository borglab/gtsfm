import abc
from typing import DefaultDict, Dict, List, Tuple

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

    def __init__(self, calibration, global_poses: List[gtsam.Pose3]) -> None:
        self.__calibration = calibration
        self.__global_pose_list = global_poses
    




