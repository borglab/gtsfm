import abc
from typing import Dict, List, Tuple, Optional

import cv2
import dask
import gtsam
import numpy as np

from data_association.feature_tracks import FeatureTrackGenerator

class DataAssociation(FeatureTrackGenerator):
    """
    Class to form feature tracks; for each track, call LandmarkInitialization
    """
    def __init__(self) -> None:
        """
        Args:
            matches: Dict of pairwise matches of form {(img1, img2): (features1, features2)}
            global poses: list of poses  
            calibrationFlag: flag to set shared or individual calibration
            calibration: shared calibration
            camera_list: list of individual cameras (if calibration not shared)
            feature_list: List of features in each image
        """
        

    def run(self, 
        matches: Dict[Tuple[int, int], Tuple[int, int]], 
        global_poses: List[gtsam.Pose3], 
        calibrationFlag: bool, 
        reprojection_threshold: float,
        min_track_length: int,
        use_ransac: bool,
        calibration: gtsam.Cal3Bundler, 
        camera_list: List,
        feature_list: List[List[np.ndarray]]) -> List:
        
        super().__init__(matches, feature_list)
        triangulated_landmark_map = []        
        sfmdata_landmark_map = self.filtered_landmark_data
        # point indices are represented as j
        # nb of 3D points = nb of tracks, hence track_idx represented as j
        for j in range(len(sfmdata_landmark_map)):
            # if self.calibrationFlag == True:
            LMI = LandmarkInitialization(calibrationFlag, reprojection_threshold, use_ransac, calibration, global_poses, camera_list)
            # else:
                # LMI = LandmarkInitialization(calibrationFlag, sfmdata_landmark_map[j], camera_list)
            triangulated_data = LMI.triangulate(sfmdata_landmark_map[j])
            filtered_track = LMI.filter_reprojection_error(triangulated_data)
            if filtered_track.number_measurements() >= min_track_length:
                triangulated_landmark_map.append(filtered_track)
            else:
                print("Track length < {} discarded".format(min_track_length))
        return triangulated_landmark_map

    def create_computation_graph(self,
        matches: Dict[Tuple[int, int], Tuple[int, int]], 
        global_poses: List[gtsam.Pose3], 
        calibrationFlag: bool, 
        reprojection_threshold: float,
        min_track_length: int,
        use_ransac: bool,
        calibration: gtsam.Cal3Bundler, 
        camera_list: List,
        feature_list: List[List]):
        
        return dask.delayed(self.run)(matches, global_poses, calibrationFlag, reprojection_threshold, min_track_length, use_ransac, calibration, camera_list, feature_list)


class LandmarkInitialization():
    """
    Class to initialize landmark points via triangulation
    """

    def __init__(self, 
        calibrationFlag: bool,
        reprojection_threshold: float,
        use_ransac: bool,
        calibration: Optional[gtsam.Cal3_S2] = None, 
        track_poses: Optional[List[gtsam.Pose3]] = None, 
        track_cameras: Optional[List[gtsam.Cal3_S2]] = None,   
    ) -> None:
        """
        Args:
            calibrationFlag: check if shared calibration exists(True) or each camera has individual calibration(False)
            obs_list: Feature track of type [(img_idx, img_measurement),..]
            calibration: Shared calibration
            track_poses: List of poses
            track_cameras: List of cameras
        """
        self.sharedCal_Flag = calibrationFlag
        self.threshold = reprojection_threshold
        self.calibration = calibration
        # for shared calibration
        if track_poses is not None:
            self.track_pose_list = track_poses
        # for multiple cameras with individual calibrations
        if track_cameras is not None:
            self.track_camera_list = track_cameras
    

    def extract_end_measurements(self, track: List[Tuple]) -> Tuple[gtsam.Pose3Vector, List, gtsam.Point2Vector]:
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
        for k in range(len(track)):
            img_idx, img_Pt = track[k]
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


    def triangulate(self, track: List[Tuple]) -> Dict:
        """
        Args:
            track: feature track
        Returns:
            triangulated_landmark: triangulated landmark
        """
        pose_estimates, camera_values, img_measurements = self.extract_end_measurements(track)
        triangulated_track = dict()
        optimize = True
        rank_tol = 1e-9
        # if shared calibration provided for all cameras
        if self.sharedCal_Flag:
            if not pose_estimates:
                raise Exception('track_poses arg or pose estimates missing')
            triangulated_pt = gtsam.triangulatePoint3(pose_estimates, self.calibration, img_measurements, rank_tol, optimize)
            triangulated_track.update({tuple(triangulated_pt) : track})
        else:
            if not camera_values:
                raise Exception('track_cameras arg or camera values missing')
            triangulated_pt = gtsam.triangulatePoint3(camera_values, img_measurements, rank_tol, optimize)
            triangulated_track.update({tuple(triangulated_pt) : track})
        return triangulated_track
    
    def filter_reprojection_error(self, triangulated_track: Dict):
        """
        Filter measurements that have high reprojection error in a camera
        Args:
            Triangulated track, with triangulated pt as key and track as value
        Returns:
            SfmTrack object
        """
        # TODO: Set threshold = 5*smallest_error in track?
        # threshold = 5
        new_track = gtsam.SfmTrack(list(triangulated_track.keys())[0])
        
        # measurement_idx represented as k
        for triangulated_pt, track in triangulated_track.items():
            for (i, measurement) in track:
                if self.sharedCal_Flag:
                    camera = gtsam.PinholeCameraCal3Bundler(self.track_pose_list[i], self.calibration)
                else:
                    camera = gtsam.PinholeCameraCal3Bundler(self.track_pose_list[i], self.track_camera_list[i])
                # Project to camera 1
                uc = camera.project(triangulated_pt)[0]
                vc = camera.project(triangulated_pt)[1]
                # Projection error in camera
                error = (uc - measurement[0])**2 + (vc - measurement[1])**2
                if error < self.threshold:
                    new_track.add_measurement(i, measurement)
        return new_track

        
        