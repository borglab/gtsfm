import abc
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple, Optional

import numpy as np
import cv2
import dask

import gtsam
from data_association.feature_tracks import FeatureTrackGenerator

from enum import Enum

class triangulation_params(Enum):
    UNIFORM = 1
    BASELINE = 2
    MAX_TO_MIN = 3

class LandmarkInitialization(metaclass=abc.ABCMeta):
    """
    Class to initialize landmark points via triangulation w or w/o RANSAC inlier/outlier selection

    triangulate(
        track: List, 
        use_ransac: bool,
        sampling_method: Optional[triangulation_params] = None, 
        num_samples: Optional[int] = None, 
        thresh: Optional[float] = None
    ) -> Dict:
    
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
        
        # for shared calibration
        if calibration is not None:
            self.calibration = calibration
        if track_poses is not None:
            self.track_pose_list = track_poses
        
        # for multiple cameras with individual calibrations
        if track_cameras is not None:
            self.track_camera_list = track_cameras
    
        def triangulate(
        self, track: List, 
        use_ransac: bool,
        sampling_method: Optional[triangulation_params] = None, 
        num_samples: Optional[int] = None, 
        thresh: Optional[float] = None
        ) -> Dict:
        """
        Triangulation in RANSAC loop
        Args:
            track: feature track from which measurements are to be extracted
            use_ransac: a tag to enable/disable the RANSAC based triangulation
            sampling_method (optional): triangulation_params.UNIFORM    -> sampling uniformly, 
                                        triangulation_params.BASELINE   -> sampling based on estimated baseline, 
                                        triangulation_params.MAX_TO_MIN -> sampling from max to min 
            num_samples (optional): desired number of samples
            tresh (optional): threshold for RANSAC inlier filtering
        Returns:
            triangulated_landmark: triangulated landmark
        """
        if use_ransac:
            # Generate all possible matches
            matches = self.generate_matches(track)
            
            # Initialize scores for all matches
            scores = self.initialize_scores(track, matches, sampling_method)
            
            # Check the validity of num_samples  
            if  num_samples > len(matches):
                num_samples = len(matches)
            
            # Sampling
            samples = self.sampling(scores, sampling_method, num_samples)
            
            # Initialize the best output containers
            best_pt = gtsam.Point3()
            best_votes = 0
            best_error = 1e10
            best_inliers = []

            for s in range(num_samples):
                k1, k2 = matches[samples[s]]

                idx1, pt1 = track[k1]
                idx2, pt2 = track[k2]
                
                if self.sharedCal_Flag:
                    pose_estimates = [self.track_pose_list[idx1], self.track_pose_list[idx2]]
                else:
                    camera_estimates = [self.track_camera_list[idx1], self.track_camera_list[idx2]]

                img_measurements = [pt1,pt2]
                
                # if shared calibration provided for all cameras
                if self.sharedCal_Flag:
                    triangulated_pt = gtsam.triangulatePoint3(pose_estimates, self.calibration, img_measurements, rank_tol=1e-9, optimize=True)
                else:
                    triangulated_pt = gtsam.triangulatePoint3(camera_estimates, img_measurements, rank_tol=1e-9, optimize=True)
                    
                errors = self.reprojection_error(triangulated_pt, track)
                votes = [err < thresh for err in errors]

                sum_error = sum(errors)
                sum_votes = sum([int(v) for v in votes])

                if (
                    (sum_votes > best_votes) or 
                    (sum_votes == best_votes and best_error > sum_error)
                ):
                    best_votes = sum_votes 
                    best_error = sum_error
                    best_pt = triangulated_pt
                    best_inliers = votes
        else:
            best_inliers = [True for k in range(len(track))]
        
        pose_track, camera_track, measurement_track = self.extract_measurements(track, best_inliers)
        
        triangulated_track = dict()
        
        # if shared calibration provided for all cameras
        if self.sharedCal_Flag:
            triangulated_pt_track = gtsam.triangulatePoint3(pose_track, self.calibration, measurement_track, rank_tol, optimize=True)
            triangulated_track.update({tuple(triangulated_pt_track) : track})
        else:
            triangulated_pt_track = gtsam.triangulatePoint3(camera_track, measurement_track, rank_tol, optimize=True)
            triangulated_track.update({tuple(triangulated_pt_track) : track})
        
        # we may want to compare the initialized best_pt with triangulated_pt_track
        # if use_ransac:
        #    error_check = (best_pt - triangulated_pt_track).norm()
        return triangulated_track

    def generate_matches(self, track: List) -> List[Tuple[int,int]]:
        """
        Extract all possible measurement pairs (k1, k2) in a track for triangulation.
        Args:
            track: feature track from which measurements are to be extracted

        Returns:
            matches: all possible matches in a given track 
        """
        matches = []

        for k1 in range(len(track)):
            for k2 in range(k1+1,len(track)):
                matches.append([k1,k2])
                
        return matches
    
    def initialize_scores(self, track: List, matches: List, sampling_method: triangulation_params) -> List[float]:
        """
        Extract all possible measurement pairs (k1, k2) and their scores in a track for triangulation.
        Args:
            track: feature track from which measurements are to be extracted
            matches: all possible matches in a given track
            sampling_method: triangulation_params.UNIFORM    -> sampling uniformly, 
                             triangulation_params.BASELINE   -> sampling based on estimated baseline, 
                             triangulation_params.MAX_TO_MIN -> sampling from max to min 
        Returns:
            scores: scores of corresponding matches for RANSAC sampling 
        """

        scores = []

        for k in range(len(matches)):
            k1, k2 = matches[k]
            
            if (
                sampling_method == triangulation_params.UNIFORM
            ):
                scores.append(1.0)

            if (
                sampling_method == triangulation_params.BASELINE or 
                sampling_method == triangulation_params.MAX_TO_MIN
            ):
                idx1, = track[k1]
                idx2, = track[k2]
                if self.sharedCal_Flag:
                    p1 = self.track_pose_list[idx1]
                    p2 = self.track_pose_list[idx2]
                    # it is not a very correct approximation of depth, will do it better later
                    l2_distance = (p1.inverse()*p2).translation().norm() 
                    scores.append(l2_distance)
                else:
                    scores.append(1.0)
            
        return scores

    def sampling(self, scores: List, sampling_method: triangulation_params, num_samples: int) -> List[int]:
        """
        Generate a list of matches for triangulation 
        Args:
            scores: scores for sampling
            sampling_method: triangulation_params.UNIFORM    -> sampling uniformly, 
                             triangulation_params.BASELINE   -> sampling based on estimated baseline, 
                             triangulation_params.MAX_TO_MIN -> sampling from max to min 
            num_samples: desired number of samples
        Returns:
            indexes of matches: index of selected match
        """
        if sum(scores) <= 0.0:
            raise Exception("Sum of scores cannot be Zero (or smaller than Zero)! It must a bug somewhere")
        
        sample_index = []

        for s in range(num_samples):
            if (
                sampling_method == triangulation_params.UNIFORM or
                sampling_method == triangulation_params.BASELINE
            ): 
                # normalization of scores for sampling
                nscores = [s/sum(scores) for s in scores]
                
                # generate random number from 0 to 1
                random_sampler = np.random.uniform()

                # sampling
                acc = 0.0
                for idx in range(len(nscores)):
                    acc += nscores[idx]
                    if (acc > random_sampler):
                        # will never sample it again
                        scores[idx] = 0.0
                        #  push the index
                        sample_index.append(idx)
            
            if (
                sampling_method == triangulation_params.MAX_TO_MIN
            ): 
                # calculate the index of the maximum element
                idx = scores.index(max(scores))
                # will never sample it again
                scores[idx] = 0.0
                sample_index.append(idx)
        
        return sample_index

    def reprojection_error(self, triangulated_pt: gtsam.Point3, track: List) -> List[float]:
        """
        Calculate average reprojection error in a given track
        Args:
            triangulated point: triangulated 3D point 
            track: the measurements of a track
        Returns:
            reprojection errors 
        """
        errors = []
        
        for (i, measurement) in track:
            if self.sharedCal_Flag:
                camera = gtsam.PinholeCameraCal3Bundler(self.track_pose_list[i], self.calibration)
            else:
                camera = gtsam.PinholeCameraCal3Bundler(self.track_pose_list[i], self.track_camera_list[i])
            # Project to camera 1
            uc = camera.project(triangulated_pt)[0]
            vc = camera.project(triangulated_pt)[1]
            # Projection error in camera
            errors.append((uc - measurement[0])**2 + (vc - measurement[1])**2)
        
        return errors

    def extract_measurements(self, track: List, inliers: List) -> Tuple[gtsam.Pose3Vector, List, gtsam.Point2Vector]:
        """
        Extract measurements in a track for triangulation.
        Args:
            track: feature track from which measurements are to be extracted
            inliers: a boolean list that indicates the validity of each measurements
        Returns:
            pose_track: Poses of first and last measurements in track
            camera_track: Individual camera calibrations for first and last measurement
            measurement_track: Observations corresponding to first and last measurements
        """
        
        pose_track = gtsam.Pose3Vector()
        camera_track = []
        measurement_track = gtsam.Point2Vector()

        for k in range(len(track)):
            if inliers[k]:
                img_idx, img_Pt = track[k]
                if self.sharedCal_Flag:
                    pose_track.append(self.track_pose_list[img_idx])
                else:
                    camera_track.append(self.track_camera_list[img_idx])
                
                measurement_track.append(img_Pt)
        
        if (
            len(pose_track) < 2 or 
            len(camera_track) < 2 or 
            len(measurement_track) < 2
        ):
            raise Exception("Nb of measurements should not be > 2. \
                Number of poses is: {}, number of cameras is: {} and number of observations is {}".format(
                    len(pose_track), 
                    len(camera_track), 
                    len(measurement_track)))
        
        return pose_track, camera_track, measurement_track

