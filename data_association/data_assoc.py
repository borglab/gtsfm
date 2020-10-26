import abc
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple, Optional

import numpy as np
import cv2

import gtsam
from data_association.feature_tracks import FeatureTrackGenerator

class DataAssociation(FeatureTrackGenerator):
    """
    Class to form feature tracks; for each track, call LandmarkInitialization
    """
    def __init__(self) -> None:
        """
        #CAN NUM POSES BE REPLACED WITH LEN(POSES)?
        Args:
            matches: Dict of pairwise matches of form {(img1, img2): (features1, features2)
            num_poses: number of poses
            global poses: list of poses  
            calibrationFlag: flag to set shared or individual calibration
            calibration: shared calibration
            camera_list: list of individual cameras (if calibration not shared)
            feature_list: List of features in each image
        """
        

    def run(self, 
        matches: Dict[Tuple[int, int], Tuple[int, int]], 
        num_poses: int, 
        global_poses: List[gtsam.Pose3], 
        calibrationFlag: bool, 
        calibration: gtsam.Cal3_S2, 
        camera_list: List,
        feature_list: List[List]) -> List:
        
        self.calibrationFlag = calibrationFlag
        self.calibration = calibration
        self.features_list = feature_list
        super().__init__(matches, num_poses, feature_list)
        triangulated_landmark_map = []        
        sfmdata_landmark_map = self.filtered_landmark_data
        for track_idx in range(len(sfmdata_landmark_map)):
            if self.calibrationFlag == True:
                LMI = LandmarkInitialization(calibrationFlag, sfmdata_landmark_map[track_idx], calibration,global_poses)
            else:
                LMI = LandmarkInitialization(calibrationFlag, sfmdata_landmark_map[track_idx], camera_list)
            triangulated_data = LMI.triangulate(sfmdata_landmark_map[track_idx])
            filtered_track = LMI.filter_reprojection_error(triangulated_data.point3(), sfmdata_landmark_map[track_idx])
            if filtered_track.number_measurements() > 2:
                triangulated_landmark_map.append(filtered_track)
            else:
                print("Track length < 3 discarded")
        return triangulated_landmark_map

    def create_computation_graph(self,
        matches: Dict[Tuple[int, int], Tuple[int, int]], 
        num_poses: int, 
        global_poses: List[gtsam.Pose3], 
        calibrationFlag: bool, 
        calibration: gtsam.Cal3_S2, 
        camera_list: List,
        feature_list: List[List]):
        
        return dask.delayed(self.run)(matches, num_poses, global_poses, calibrationFlag, calibration, camera_list, feature_list)


class LandmarkInitialization(metaclass=abc.ABCMeta):
    """
    Class to initialize landmark points via triangulation
    """

    def __init__(
        self, 
    calibrationFlag: bool,
    obs_list: List,
    calibration: Optional[gtsam.Cal3_S2] = None, 
    track_poses: Optional[List[gtsam.Pose3]] = None, 
    track_cameras: Optional[List[gtsam.Cal3_S2]] = None
    ) -> None:
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
        self.calibration = calibration
        # for shared calibration
        if track_poses is not None:
            self.track_pose_list = track_poses
        # for multiple cameras with individual calibrations
        if track_cameras is not None:
            self.track_camera_list = track_cameras
    
    
    def create_landmark_map(self, filtered_map:gtsam.SfmData, triangulated_pts: List) -> Dict:
        landmark_map = filtered_map.copy()
        for idx, (key, val) in enumerate(filtered_map.items()):
            new_key = tuple(triangulated_pts[idx])
            # copy the value
            landmark_map[new_key] = filtered_map[key]
            del landmark_map[key]
        return landmark_map

    def extract_end_measurements(self, track: gtsam.SfmTrack) -> Tuple[gtsam.Pose3Vector, List, gtsam.Point2Vector]:
        """
        Extract first and last measurements in a track for triangulation.
        Args:
            track: feature track from which measurements are to be extracted
        Returns:
            pose_estimates: Poses of first and last measurements in track
            camera_list: Individual camera calibrations for first and last measurement
            img_measurements: Observations corresponding to first and last measurements
        """
        pose_estimates_track = gtsam.Pose3Vector()
        pose_estimates = gtsam.Pose3Vector()
        cameras_list_track = []
        cameras_list = []
        img_measurements_track = gtsam.Point2Vector()
        img_measurements = gtsam.Point2Vector()
        for measurement_idx in range(track.number_measurements()):
            img_idx, img_Pt = track.measurement(measurement_idx)
            if self.sharedCal_Flag:
                pose_estimates_track.append(self.track_pose_list[img_idx])
            else:
                cameras_list_track.append(self.track_camera_list[img_idx]) 
            img_measurements_track.append(img_Pt)
        if pose_estimates_track:
            pose_estimates.append(pose_estimates_track[0]) 
            pose_estimates.append(pose_estimates_track[-1])
        else:
            cameras_list = [cameras_list_track[0], cameras_list_track[-1]]
        img_measurements.append(img_measurements_track[0])
        img_measurements.append(img_measurements_track[-1])

        if len(pose_estimates) > 2 or len(cameras_list) > 2 or len(img_measurements) > 2:
            raise Exception("Nb of measurements should not be > 2. \
                Number of poses is: {}, number of cameras is: {} and number of observations is {}".format(
                    len(pose_estimates), 
                    len(cameras_list), 
                    len(img_measurements)))
        
        return pose_estimates, cameras_list, img_measurements


    def triangulate(self, track: gtsam.SfmTrack) -> gtsam.SfmTrack:
        """
        Args:
            track: feature track
        Returns:
            triangulated_landmark: triangulated landmark
        """

        pose_estimates, camera_values, img_measurements = self.extract_end_measurements(track)
        optimize = True
        rank_tol = 1e-9
        # if shared calibration provided for all cameras
        if self.sharedCal_Flag:
            if self.track_pose_list == None or not pose_estimates:
                raise Exception('track_poses arg or pose estimates missing')
            triangulated_pt = gtsam.triangulatePoint3(pose_estimates, self.calibration, img_measurements, rank_tol, optimize)
            track.set_point3(triangulated_pt)
        else:
            if self.track_camera_list == None or not camera_values:
                raise Exception('track_cameras arg or camera values missing')
            triangulated_pt = gtsam.triangulatePoint3(camera_values, img_measurements, rank_tol, optimize)
            track.set_point3(triangulated_pt)
        return track
    
    def filter_reprojection_error(self, triangulated_pt: gtsam.Point3,track: gtsam.SfmTrack):
        # TODO: Set threshold = 5*smallest_error in track?
        threshold = 5
        new_track = gtsam.SfmTrack()
        
        for measurement_idx in range(track.number_measurements()):
            pose_idx, measurement = track.measurement(measurement_idx)
            if self.sharedCal_Flag:
                camera = gtsam.PinholeCameraCal3_S2(self.track_pose_list[pose_idx], self.calibration)
            else:
                camera = gtsam.PinholeCameraCal3_S2(self.track_pose_list[pose_idx], self.track_camera_list[pose_idx])
            # Project to camera 1
            uc = camera.project(triangulated_pt)[0]
            vc = camera.project(triangulated_pt)[1]
            # Projection error in camera
            error = (uc - measurement[0])**2 + (vc - measurement[1])**2
            if error < threshold:
                new_track.add_measurement((pose_idx, measurement))
                new_track.set_point3(triangulated_pt)
        return new_track

        
        