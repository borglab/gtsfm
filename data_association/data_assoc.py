""" Create 2D-3D data association as a precursor to Bundle Adjustment.
1. Forms feature tracks from verified correspondences and global poses.
2. Triangulates 3D world points for each track (Ransac and simple triangulation modes available)
3. Filters tracks based on reprojection error.

Authors: Sushmita Warrier, Xiaolong Wu
"""

import dask
import gtsam
import numpy as np

from common.keypoints import Keypoints
from data_association.feature_tracks import FeatureTrackGenerator
from enum import Enum
from typing import Dict, List, Tuple, Optional

class TriangulationParam(Enum):
    UNIFORM = 1
    BASELINE = 2
    MAX_TO_MIN = 3
    SIMPLE = 4

class DataAssociation(FeatureTrackGenerator):
    """ Class to form feature tracks; for each track, call LandmarkInitialization.
    """
    def __init__(self, matches: Dict[Tuple[int, int], np.ndarray], feature_list: List[Keypoints]) -> None:
        """ Form feature tracks.

        Args:
            matches: Dict of pairwise matches of form {(img1_idx, img2_idx): np.array()}.
                The array is of shape Nx2; N being the nb of features and each row being (feature_idx1, idx2).
            feature_list: List of keypoints.
        """
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
        """ Triangulate and filter points for feature tracks.

        Args:
            global poses: list of poses.
            sharedcalibrationFlag: flag to set shared or individual calibration
            reprojection_threshold: error threshold for track filtering.
            min_track_length: Minimum nb of views that must support a landmark for it to be accepted.
            use_ransac: Select between simple triangulation(False) and ransac-based triangulation(True).
            calibration: shared calibration.
            camera_list: list of individual cameras (if calibration not shared).

        Returns:
            List of SfmTrack objects, containing feature tracks and their 3D landmark points.
        """        
        
        triangulated_landmark_map = []        
        sfmdata_landmark_map = self.filtered_landmark_data
        # point indices are represented as j
        # nb of 3D points = nb of tracks, hence track_idx represented as j
        for j in range(len(sfmdata_landmark_map)):
            LMI = LandmarkInitialization(sharedcalibrationFlag, reprojection_threshold, global_poses, calibration, camera_list)
            triangulated_data = LMI.triangulate(sfmdata_landmark_map[j], use_ransac)
            filtered_track = LMI.filter_reprojection_error(triangulated_data)

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


class LandmarkInitialization():
    """
    Class to initialize landmark points via triangulation.
    """

    def __init__(self, 
        sharedcalibrationFlag: bool,
        reprojection_threshold: float,
        track_poses: List[gtsam.Pose3], 
        calibration: Optional[gtsam.Cal3Bundler] = None, 
        track_cameras: Optional[gtsam.CameraSetCal3Bundler] = None,   
    ) -> None:
        """
        Args:
            sharedcalibrationFlag: check if shared calibration exists(True) or each camera has individual calibration(False).
            obs_list: Feature track of type [(camera_idx, img_measurement),..].
            calibration: Shared calibration of type Cal3Bundler.
            track_poses: List of poses.
            track_cameras: List of individual cameras, if not using shared calibration.
        """
        self.sharedCal_Flag = sharedcalibrationFlag
        self.threshold = reprojection_threshold
        self.calibration = calibration
        # for shared calibration
        if track_poses is None:
            raise Exception("Poses required")
        self.track_pose_list = track_poses
        # for multiple cameras with individual calibrations
        if track_cameras is not None:
            self.track_camera_list = track_cameras
    

    def extract_end_measurements(self, track: List[Tuple]) -> Tuple[gtsam.Pose3Vector, List, gtsam.Point2Vector]:
        """
        Extract first and last measurements in a track for triangulation.

        Args:
            track: feature track from which measurements are to be extracted.

        Returns:
            pose_estimates: Poses of first and last measurements in track.
            camera_list: Individual camera calibrations for first and last measurement.
            img_measurements: Observations corresponding to first and last measurements.
        """
        pose_estimates_track = gtsam.Pose3Vector()
        pose_estimates = gtsam.Pose3Vector()
        cameras_list_track = gtsam.CameraSetCal3Bundler()
        cameras_list = gtsam.CameraSetCal3Bundler()
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
            cameras_list.append(cameras_list_track[0])
            cameras_list.append(cameras_list_track[-1])

        img_measurements.append(img_measurements_track[0])
        img_measurements.append(img_measurements_track[-1])

        if len(pose_estimates) > 2 or len(cameras_list) > 2 or len(img_measurements) > 2:
            raise Exception("Nb of measurements should not be > 2. \
                Number of poses is: {}, number of cameras is: {} and number of observations is {}".format(
                    len(pose_estimates), 
                    len(cameras_list), 
                    len(img_measurements)))
        
        return pose_estimates, cameras_list, img_measurements


    def triangulate(self, track: List[Tuple], use_ransac: bool) -> Dict:
        """ Triangulate based on a simple algorithm taking largest baseline, assumed to be endpoints of a track.

        Args:
            track: Feature track as list of (camera_idx,measurements).

        Returns:
            Feature track as a dict with landmark as key and track as value.
        """
        triangulated_track = dict()
        if use_ransac:
            pass
        if not use_ransac:
            pose_estimates, camera_values, img_measurements = self.extract_end_measurements(track)
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
        Filter measurements that have high reprojection error in a camera.

        Args:
            Triangulated track: Dict with triangulated pt as key and track as value.

        Returns:
            SfmTrack object with filtered track.
        """
        new_track = gtsam.SfmTrack(list(triangulated_track.keys())[0])
        
        # measurement_idx represented as k
        for triangulated_pt, track in triangulated_track.items():
            for (i, measurement) in track:
                if self.sharedCal_Flag:
                    camera = gtsam.PinholeCameraCal3Bundler(self.track_pose_list[i], self.calibration)
                else:
                    camera = self.track_camera_list[i]
                # Project to camera 1
                uc = camera.project(triangulated_pt)[0]
                vc = camera.project(triangulated_pt)[1]
                # Projection error in camera
                error = (uc - measurement[0])**2 + (vc - measurement[1])**2
                if error < self.threshold:
                    new_track.add_measurement(i, measurement)
        return new_track

        
        