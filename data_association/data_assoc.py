""" Create 2D-3D data association as a precursor to Bundle Adjustment.
1. Forms feature tracks from verified correspondences and global poses.
2. Triangulates 3D world points for each track (Ransac and simple triangulation modes available)
3. Filters tracks based on reprojection error.

Authors: Sushmita Warrier, Xiaolong Wu
"""

import dask
from dask.delayed import Delayed
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
        sharedcalibrationFlag: bool, 
        min_track_length: int,
        use_ransac: bool,
        calibration: Optional[gtsam.Cal3Bundler] = None, 
        global_poses: Optional[List[gtsam.Pose3]] = None, 
        camera_list: Optional[List[gtsam.PinholeCameraCal3Bundler]] = None,
        sampling_method: Optional[TriangulationParam] = None,
        num_samples: Optional[int] = None,
        reprojection_threshold: Optional[float] = None
    ) -> List:
        """ The main purpose of data asscociation is to triangulate points and perform inlier/outlier detection
        
        Args:
            sharedcalibrationFlag: flag to set shared (calibration + global_poses)or individual calibration (camera list)
            min_track_length: the minmimum length of track
            use_ransac: flag to set the usage of ransac-based triangulation or not
            sampling_method: sampling method for ransac-based triangulation
            num_samples: number of samples in ransac-based triangulation
            reprojection_threshold: the maximum reprojection distance that can be seen as inliers
            calibration(optional): shared calibration
            global poses(optional): list of poses  
            camera_list(optional): list of individual cameras (if calibration not shared)
            
        """
        triangulated_landmark_map = []        
        sfmdata_landmark_map = self.filtered_landmark_data
        
        # point indices are represented as j
        # nb of 3D points = nb of tracks, hence track_idx represented as j
        LMI = LandmarkInitialization(sharedcalibrationFlag, calibration, global_poses, camera_list)
        
        for j in range(len(sfmdata_landmark_map)):
            filtered_track = LMI.triangulate(sfmdata_landmark_map[j], use_ransac, sampling_method, num_samples, reprojection_threshold)

            if filtered_track.number_measurements() >= min_track_length:
                triangulated_landmark_map.append(filtered_track)
            else:
                print("Track length {} < {} discarded".format(
                    filtered_track.number_measurements(), 
                    min_track_length)
                )
        
        return triangulated_landmark_map

    def create_computation_graph(self,
        sharedcalibrationFlag: bool, 
        min_track_length: int,
        use_ransac: bool,
        calibration: Optional[gtsam.Cal3Bundler] = None, 
        global_poses: Optional[List[gtsam.Pose3]] = None, 
        camera_list: Optional[List[gtsam.PinholeCameraCal3Bundler]] = None,
        sampling_method: Optional[TriangulationParam] = None,
        num_samples: Optional[int] = None,
        reprojection_threshold: Optional[float] = None
    ) -> List:
        
        return dask.delayed(self.run)(
            sharedcalibrationFlag, 
            min_track_length,
            use_ransac,
            calibration, 
            global_poses, 
            camera_list,
            sampling_method,
            num_samples,
            reprojection_threshold
        )

class LandmarkInitialization():
    """
    Class to initialize landmark points via triangulation w or w/o RANSAC inlier/outlier selection

    triangulate(
        track: List, 
        use_ransac: bool,
        sampling_method: Optional[TriangulationParam] = None, 
        num_samples: Optional[int] = None, 
        thresh: Optional[float] = None
    ) -> Dict:
    
    """

    def __init__(
        self, 
        sharedcalibrationFlag: bool,
        calibration: Optional[gtsam.PinholeCameraCal3Bundler] = None, 
        track_poses: Optional[List[gtsam.Pose3]] = None, 
        track_cameras: Optional[List[gtsam.PinholeCameraCal3Bundler]] = None
        ) -> None:
        """
        Args:
            sharedcalibrationFlag: check if shared calibration exists(True) or each camera has individual calibration(False)
            calibration: Shared calibration
            track_poses: List of poses in a feature track
            track_cameras: List of cameras in a feature track
        """
        self.sharedCal_Flag = sharedcalibrationFlag
        
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
        sampling_method: Optional[TriangulationParam] = None, 
        num_samples: Optional[int] = None, 
        thresh: Optional[float] = None
    ) -> gtsam.SfmTrack:
        """
        Triangulation in RANSAC loop
        Args:
            track: feature track from which measurements are to be extracted
            use_ransac: a tag to enable/disable the RANSAC based triangulation
            sampling_method (optional): TriangulationParam.UNIFORM    -> sampling uniformly, 
                                        TriangulationParam.BASELINE   -> sampling based on estimated baseline, 
                                        TriangulationParam.MAX_TO_MIN -> sampling from max to min 
            num_samples (optional): desired number of samples
            tresh (optional): threshold for RANSAC inlier filtering
        Returns:
            triangulated_track: triangulated track after triangulation with measurements
        """
        if use_ransac:
            # Generate all possible matches
            matches = self.generate_matches(track)
                        
            # Check the validity of num_samples  
            if  num_samples > len(matches):
                num_samples = len(matches)
            
            # Sampling
            samples = self.sampling(track, matches, sampling_method, num_samples)
            
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
                    pose_estimates = gtsam.Pose3Vector()
                    pose_estimates.append(self.track_pose_list[idx1])
                    pose_estimates.append(self.track_pose_list[idx2])
                else:
                    camera_estimates = [self.track_camera_list[idx1], self.track_camera_list[idx2]]

                img_measurements = gtsam.Point2Vector()
                img_measurements.append(pt1)
                img_measurements.append(pt2)
                
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
            triangulated_pt_track = gtsam.triangulatePoint3(pose_track, self.calibration, measurement_track, rank_tol=1e-9, optimize=True)
            triangulated_track.update({tuple(triangulated_pt_track) : track})
        else:
            triangulated_pt_track = gtsam.triangulatePoint3(camera_track, measurement_track, rank_tol=1e-9, optimize=True)
            triangulated_track.update({tuple(triangulated_pt_track) : track})
        
        # we may want to compare the initialized best_pt with triangulated_pt_track
        # if use_ransac:
        #    error_check = (best_pt - triangulated_pt_track).norm()
        return self.inlier_to_track(triangulated_track, best_inliers)

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
        
    def sampling(self, track: List, matches: List, sampling_method: TriangulationParam, num_samples: int) -> List[int]:
        """Generate a list of matches for triangulation 
        
        Args:
            track: feature track from which measurements are to be extracted
            matches: all possible matches in a given track
            sampling_method: TriangulationParam.UNIFORM    -> sampling uniformly, 
                             TriangulationParam.BASELINE   -> sampling based on estimated baseline, 
                             TriangulationParam.MAX_TO_MIN -> sampling from max to min 
            num_samples: desired number of samples
        Returns:
        indexes of matches: index of selected match
        """
        # Initilize scores as uniform distribution
        scores = np.ones(len(matches),dtype=float)
        
        if (
            (sampling_method == TriangulationParam.BASELINE or 
            sampling_method == TriangulationParam.MAX_TO_MIN) and 
            self.sharedCal_Flag
        ):
            for k in range(len(matches)):
                k1, k2 = matches[k]

                idx1, pt1 = track[k1]
                idx2, pt2 = track[k2]

                wTc1 = gtsam.Pose3(self.track_pose_list[idx1])
                wTc2 = gtsam.Pose3(self.track_pose_list[idx2])
                
                # it is not a very correct approximation of depth, will do it better later
                scores[k] = np.linalg.norm(wTc1.compose(wTc2.inverse()).translation()) 
        
        # Check the validity of scores
        if sum(scores) <= 0.0:
            raise Exception("Sum of scores cannot be Zero (or smaller than Zero)! It must a bug somewhere")
        
        if (
            sampling_method == TriangulationParam.UNIFORM or
            sampling_method == TriangulationParam.BASELINE
        ): 
            sample_index = np.random.choice(len(scores), num_samples, replace=False, p=scores/scores.sum())
                
        if (
            sampling_method == TriangulationParam.MAX_TO_MIN
        ): 
            sample_index = np.argsort(scores)[-num_samples:]
        
        return sample_index.tolist()

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

        if self.sharedCal_Flag:
            if ( 
                len(pose_track) < 2 or 
                len(measurement_track) < 2
            ):
                raise Exception("Nb of measurements should not be <= 2. \
                    Number of poses is: {} and number of observations is {}".format(
                        len(pose_track), 
                        len(measurement_track)))
        else:
            if ( 
                len(camera_track) < 2 or 
                len(measurement_track) < 2
            ):
                raise Exception("Nb of measurements should not be <= 2. \
                     number of cameras is: {} and number of observations is {}".format(
                        len(camera_track), 
                        len(measurement_track)))

        return pose_track, camera_track, measurement_track

    def inlier_to_track(self, triangulated_track: dict, inlier:list) -> gtsam.SfmTrack:
        """
        Generate track based on inliers
        
        Args:
            triangulated_track: with triangulated pt as key and track as value
            inlier: best inlier list from ransac or all points
        Returns:
            SfmTrack object
        """
        new_track = gtsam.SfmTrack(list(triangulated_track.keys())[0])
        
        # measurement_idx represented as k
        for triangulated_pt, track in triangulated_track.items():
            for (i, measurement) in track:
                if inlier[i]:
                    new_track.add_measurement(i, measurement)
        return new_track