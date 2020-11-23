import abc
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple, Optional

import numpy as np
import cv2
import dask

import random

import gtsam
from data_association.feature_tracks import FeatureTrackGenerator

class LandmarkInitialization(metaclass=abc.ABCMeta):
    """
    Class to initialize landmark points via triangulation
    """

    """
    ROBUSTIFIED TRIANGULARIZATION: 
        It is a triangularization algirhthm in a RANSAC Loop.
    PROBABILISTIC MODEL: 
        e_z = (z^2 / b*f) * e_m 
        where e_z is the depth error, z is depth, b is baseline, f is focal length, and e_m is matching error.
        fix z, f, and e_m which is the same for all measurements in a track, 
        e_z prop 1/b
    ALGORITHM FLOWWORK: 
        1.1. Randomly select measurement pairs uniformly
        or
        1.2. Randomly select measurement pairs based the probability of estimated depth uncertainty (baseline of two camera poses)
        or 
        1.3. Select measurement pairs from high to low probability of estimated depth uncertainty (baseline of two camera poses)

        2. Triangluate the 3D points and calculate the average reprojection error (as scoring metric)

        3. Select the one hold the least score and all the inliers

        4. Use all the inlier to formulate the tracks and redo the triangularization
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
    
    def generate_matches(self, track: List, mode: int) -> Tuple[List[Tuple[int,int]], List[float]]:
        """
        Extract all possible measurement pairs (k1, k2) and their scores in a track for triangulation.
        Args:
            track: feature track from which measurements are to be extracted
            mode: 1 -> uniform sampling, 
                  2 -> sampling based on estimated probability, 
                  3 -> sampling from max to min
        Returns:
            matches: all possible matches in a track 
            scores: scores of corresponding matches for RANSAC sampling 
        """
        matches = []
        scores = []
        for k1 in range(len(track)):
            for k2 in range(k1+1,len(track)):
                matches.append([k1,k2])
                if mode == 1:
                    scores.append(1.0)
                elif mode == 2 or mode == 3:
                    idx1, = track[k1]
                    idx2, = track[k2]
                    if self.sharedCal_Flag:
                        p1 = self.track_pose_list[idx1]
                        p2 = self.track_pose_list[idx2]
                        # it is not a very correct approximation of depth, will do it better later
                        score = (p1.inverse()*p2).translation().norm() 
                        scores.append(score)
                    else:
                        scores.append(1.0)

        return matches, scores

    def sample_from_scores(self, scores: List[float], mode: int) -> Tuple[List[float], int]:
        """
        Generate a list of matches for triangularization 
        Args:
            scores: scores for sampling
            mode: 1 -> uniform sampling, 
                  2 -> sampling based on estimated probability, 
                  3 -> sampling from max to min
        Returns:
            scores: updated scores
            index: index of selected match
        """
        assert sum(scores) > 0.0

        if mode == 1 or mode == 2:
            # normalize the scores
            nscores = [s/sum(scores) for s in scores]
            
            # generate random number from 0 to 1
            random_sampler = random.uniform(0, 1)

            # sampling
            acc = 0.0
            for idx in range(len(nscores)):
                acc += nscores[idx]
                if (acc > random_sampler):
                    # will never sample it again
                    scores[idx] = 0.0
                    # return the index

                    return scores, idx
        
        elif mode == 3:
            # calculate the index of the maximum element
            idx = scores.index(max(scores))
            # will never sample it again
            scores[idx] = 0.0

            return scores, idx

    def reprojection_error(self, triangulated_pt: gtsam.Point3, track: List) -> List[float]:
        """
        Calculate average reprojection error
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
        Extract first and last measurements in a track for triangulation.
        Args:
            track: feature track from which measurements are to be extracted
        Returns:
            pose_estimates: Poses of first and last measurements in track
            camera_list: Individual camera calibrations for first and last measurement
            img_measurements: Observations corresponding to first and last measurements
        """
        
        pose_estimates_track = gtsam.Pose3Vector()
        camera_estimates_track = []
        img_measurements_track = gtsam.Point2Vector()
        for k in range(len(track)):
            if inliers[k]:
                img_idx, img_Pt = track[k]
                if self.sharedCal_Flag:
                    pose_estimates_track.append(self.track_pose_list[img_idx])
                else:
                    camera_estimates_track.append(self.track_camera_list[img_idx]) 
                img_measurements_track.append(img_Pt)
        
        if len(pose_estimates_track) > 2 or len(camera_estimates_track) > 2 or len(img_measurements_track) > 2:
            raise Exception("Nb of measurements should not be > 2. \
                Number of poses is: {}, number of cameras is: {} and number of observations is {}".format(
                    len(pose_estimates_track), 
                    len(camera_estimates_track), 
                    len(img_measurements_track)))
        
        return pose_estimates_track, camera_estimates_track, img_measurements_track


    def triangulate(self, track: List) -> Dict:
        """
        Args:
            track: feature track
        Returns:
            triangulated_landmark: triangulated landmark
        """
        pose_estimates, camera_values, img_measurements = self.extract_measurements(track)
        triangulated_track = dict()
        optimize = True
        rank_tol = 1e-9
        # if shared calibration provided for all cameras
        if self.sharedCal_Flag:
            if self.track_pose_list == None or not pose_estimates:
                raise Exception('track_poses arg or pose estimates missing')
            triangulated_pt = gtsam.triangulatePoint3(pose_estimates, self.calibration, img_measurements, rank_tol, optimize)
            triangulated_track.update({tuple(triangulated_pt) : track})
        else:
            if self.track_camera_list == None or not camera_values:
                raise Exception('track_cameras arg or camera values missing')
            triangulated_pt = gtsam.triangulatePoint3(camera_values, img_measurements, rank_tol, optimize)
            triangulated_track.update({tuple(triangulated_pt) : track})
        return triangulated_track
    
    def triangulate_ransac(self, track: List, mode: int, num_samples: int, thresh: float) -> Dict:
        """
        Triangulation in RANSAC loop
        Args:
            track: feature track from which measurements are to be extracted
            mode: 1 -> uniform sampling, 
                  2 -> sampling based on estimated probability, 
                  3 -> sampling from max to min
            num_samples: the number of samples in RANSAC loop
            tresh: threshold for RANSAC inlier filtering
        Returns:
            triangulated_landmark: triangulated landmark
        """
        
        matches, scores = self.generate_matches(track, mode)

        if len(matches) > num_samples:
            num_samples = len(matches)
        
        optimize = True
        rank_tol = 1e-9
        
        best_pt = gtsam.Point3()
        best_votes = 0
        best_inliers = []

        for s in range(num_samples):
            scores, sample = self.sample_from_scores(scores, mode)
            k1, k2 = matches[sample]

            idx1, pt1 = track[k1]
            idx2, pt2 = track[k2]
            
            if self.sharedCal_Flag:
                pose_estimates = [self.track_pose_list[idx1], self.track_pose_list[idx2]]
            else:
                camera_estimates = [self.track_camera_list[idx1], self.track_camera_list[idx2]]

            img_measurements = [pt1,pt2]
            
            # if shared calibration provided for all cameras
            if self.sharedCal_Flag:
                triangulated_pt = gtsam.triangulatePoint3(pose_estimates, self.calibration, img_measurements, rank_tol, optimize)
            else:
                triangulated_pt = gtsam.triangulatePoint3(camera_estimates, img_measurements, rank_tol, optimize)
                
            errors = self.reprojection_error(triangulated_pt, track)
            votes = [err < thresh for err in errors]

            sum_votes = sum([int(v) for v in votes])

            if sum_votes > best_votes:
                best_votes = sum_votes 
                best_pt = triangulated_pt
                best_inliers = votes
        
        pose_estimates_track, camera_estimates_track, img_measurements_track = self.extract_measurements(track, best_inliers)
        triangulated_track = dict()
        
        # if shared calibration provided for all cameras
        if self.sharedCal_Flag:
            triangulated_pt_track = gtsam.triangulatePoint3(pose_estimates_track, self.calibration, img_measurements_track, rank_tol, optimize)
            triangulated_track.update({tuple(triangulated_pt_track) : track})
        else:
            triangulated_pt = gtsam.triangulatePoint3(camera_estimates_track, img_measurements_track, rank_tol, optimize)
            triangulated_track.update({tuple(triangulated_pt_track) : track})
        
        return triangulated_track