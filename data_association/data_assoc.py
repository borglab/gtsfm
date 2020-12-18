""" Create 2D-3D data association as a precursor to Bundle Adjustment.
1. Forms feature tracks from verified correspondences and global poses.
2. Triangulates 3D world points for each track (Ransac and simple triangulation modes available)
3. Filters tracks based on reprojection error.

References: 
1. Richard I. Hartley and Peter Sturm. Triangulation. Computer Vision and Image Understanding, Vol. 68, No. 2, November, pp. 146–157, 1997
2. P. Moulon, P. Monasse. Unordered Feature Tracking Made Fast and Easy, 2012, HAL Archives.

Authors: Sushmita Warrier, Xiaolong Wu
"""
import itertools
from typing import Dict, List, NamedTuple, Optional, Tuple

import dask
from dask.delayed import Delayed
import gtsam
import numpy as np

from common.keypoints import Keypoints
from data_association.feature_tracks import FeatureTrackGenerator, SfmTrack2d
from enum import Enum
from gtsam import (
    CameraSetCal3Bundler,
    PinholeCameraCal3Bundler,
    Point3,
    Point2Vector
)

import logging

MAX_POSSIBLE_TRACK_REPROJ_ERROR = np.finfo(np.float32).max
SVD_DLT_RANK_TOL = 1e-9
NUM_SAMPLES_PER_RANSAC_HYPOTHESIS = 2

""" We specify 3 different sampling modes for robust estimation during triangulation
The fourth mode is "None", wherein all measurements are used and we assume there
are no noisy measurements. If robust estimation is requested, a pair of cameras
will be sampled """


class TriangulationParam(Enum):
    UNIFORM = 1  # sample a pair of cameras uniformly at random
    BASELINE = 2  # sample pair of cameras based on largest estimated baseline
    MAX_TO_MIN = 3  # deterministically choose hypotheses with largest estimate baseline


class DataAssociation(NamedTuple):
    """Class to form feature tracks; for each track, call LandmarkInitializer.

    Args:
        reproj_error_thresh: the maximum reprojection error allowed.
        min_track_len: min length required for valid feature track / min nb of
            supporting views required for a landmark to be valid
        sampling_method (optional): robust estimation method, specify "None" to not use RANSAC
        num_hypotheses (optional): number of samples to draw for RANSAC-based triangulation
    """

    reproj_error_thresh: float
    min_track_len: int
    sampling_method: Optional[TriangulationParam] = None
    num_hypotheses: Optional[int] = None

    def run(
        self,
        corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
        cameras: Dict[int, PinholeCameraCal3Bundler],
    ) -> gtsam.SfmData:
        """Perform data association

        Args:
            corr_idxs_dict: dictionary, with key as image pair (i1,i2) and value
                            as matching keypoint indices.
            keypoints_list: keypoints for each image.
            cameras: dictionary with image index as key, and camera object w/
                     intrinsics + extrinsics as value.

        Returns:
            SfmData
        """
        triangulated_landmark_map = gtsam.SfmData()
        tracks = FeatureTrackGenerator(corr_idxs_dict, keypoints_list)
        sfm_tracks_2d = tracks.filtered_landmark_data

        # point indices are represented as j
        # nb of 3D points = nb of tracks, hence track_idx represented as j
        point3d_initializer = Point3dInitializer(
            cameras, self.sampling_method, self.num_hypotheses, self.reproj_error_thresh
        )

        for track_2d in sfm_tracks_2d:
            filtered_track = point3d_initializer.triangulate(track_2d)

            if filtered_track.number_measurements() >= self.min_track_len:
                triangulated_landmark_map.add_track(filtered_track)
            else:
                logging.warning(
                    "Track length {} < {} discarded".format(
                        filtered_track.number_measurements(), self.min_track_len
                    )
                )
        # add cameras to landmark_map
        for cam in cameras.values():
            triangulated_landmark_map.add_camera(cam)

        return triangulated_landmark_map

    def create_computation_graph(
        self,
        corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
        cameras: Dict[int, PinholeCameraCal3Bundler],
    ) -> Delayed:
        """Creates a computation graph for performing data association.

        Args:
            corr_idxs_graph: dictionary of correspondence indices, each value
                             wrapped up as Delayed.
            keypoints_list: list of wrapped up keypoints for each image
            cameras: list of cameras wrapped up as Delayed.

        Returns:
            SfmData
        """
        return dask.delayed(self.run)(corr_idxs_dict, keypoints_list, cameras)


class Point3dInitializer(NamedTuple):
    """
    Class to initialize landmark points via triangulation w/ or w/o RANSAC inlier/outlier selection.
    We currently limit the size of each sample to 2 camera views in our RANSAC scheme.

    Args:
        track_cameras: List of cameras
        sampling_method (optional): robust estimation method. If None, robust estimation is not used
        num_hypotheses (optional): desired number of RANSAC hypotheses
        reproj_error_thresh (optional): threshold for RANSAC inlier filtering
    """

    track_camera_list: Dict[int, PinholeCameraCal3Bundler]
    sampling_method: Optional[TriangulationParam] = None
    num_hypotheses: Optional[int] = None
    reproj_error_thresh: Optional[float] = None

    def triangulate(self, track: SfmTrack2d) -> gtsam.SfmTrack:
        """
        Triangulation in a RANSAC loop.

        Args:
            track: feature track from which measurements are to be extracted

        Returns:
            SfmTrack with 3d point j and 2d measurements in multiple cameras i
        """
        if self.sampling_method:
            # Generate all possible matches
            measurement_pairs = self.generate_measurement_pairs(track)

            # We limit the number of samples to the number of actual available matches
            num_hypotheses = min(self.num_hypotheses, len(measurement_pairs))

            # Sampling
            samples = self.sample_ransac_hypotheses(
                track, measurement_pairs, num_hypotheses
            )

            # Initialize the best output containers
            best_pt = Point3()
            best_votes = 0
            best_error = MAX_POSSIBLE_TRACK_REPROJ_ERROR
            best_inliers = []

            for sample_idxs in samples:
                k1, k2 = measurement_pairs[sample_idxs]

                i1, pt1 = track.measurements[k1]
                i2, pt2 = track.measurements[k2]

                camera_estimates = CameraSetCal3Bundler()
                camera_estimates.append(self.track_camera_list.get(i1))
                camera_estimates.append(self.track_camera_list.get(i2))

                img_measurements = Point2Vector()
                img_measurements.append(pt1)
                img_measurements.append(pt2)

                # triangulate point for track
                triangulated_pt = gtsam.triangulatePoint3(
                    camera_estimates,
                    img_measurements,
                    rank_tol=SVD_DLT_RANK_TOL,
                    optimize=True,
                )

                errors = self.compute_track_reprojection_errors(triangulated_pt, track)
                # The best solution should correspond to the one with most inliers
                # If the inlier number are the same, check the average error of inliers
                votes = [err < self.reproj_error_thresh for err in errors]

                avg_error = (
                    np.array(errors) * np.array(votes).astype(float)
                ).sum() / np.array(votes).astype(float).sum()

                sum_votes = np.array(votes).astype(int).sum()

                if (sum_votes > best_votes) or (
                    sum_votes == best_votes and avg_error < best_error
                ):
                    best_votes = sum_votes
                    best_error = avg_error
                    best_pt = triangulated_pt
                    best_inliers = votes
        else:
            best_inliers = [True for _ in range(len(track.measurements))]

        camera_track, measurement_track = self.extract_measurements(track, best_inliers)

        triangulated_pt = gtsam.triangulatePoint3(
            camera_track, measurement_track, rank_tol=SVD_DLT_RANK_TOL, optimize=True
        )

        # we may want to compare the initialized best_pt with triangulated_pt_track
        return self.create_track_from_inliers(triangulated_pt, track, best_inliers)

    def generate_measurement_pairs(self, track: SfmTrack2d) -> List[Tuple[int, int]]:
        """
        Extract all possible measurement pairs in a track for triangulation.

        Args:
            track: feature track from which measurements are to be extracted

        Returns:
            measurement_idxs: all possible matching measurement indices in a given track
        """
        num_track_measurements = len(track.measurements)
        all_measurement_idxs = range(num_track_measurements)
        measurement_pair_idxs = list(
            itertools.combinations(
                all_measurement_idxs, NUM_SAMPLES_PER_RANSAC_HYPOTHESIS
            )
        )
        return measurement_pair_idxs

    def sample_ransac_hypotheses(
        self,
        track: SfmTrack2d,
        measurement_pairs: List[Tuple[int, int]],
        num_hypotheses: int,
    ) -> List[int]:
        """Generate via sampling a list of hypotheses (camera pairs) to use during triangulation

        Args:
            track: feature track from which measurements are to be extracted
            measurement_pairs: all possible indices of pairs of measurements in a given track
            num_hypotheses: desired number of samples
        Returns:
            indexes of matches: index of selected match
        """
        # Initialize scores as uniform distribution
        scores = np.ones(len(measurement_pairs), dtype=float)

        if self.sampling_method in [
            TriangulationParam.BASELINE,
            TriangulationParam.MAX_TO_MIN,
        ]:
            for k, (k1, k2) in enumerate(measurement_pairs):
                i1, pt1 = track.measurements[k1]
                i2, pt2 = track.measurements[k2]

                wTc1 = self.track_camera_list.get(i1).pose()
                wTc2 = self.track_camera_list.get(i2).pose()

                # rough approximation approximation of baseline between the 2 cameras
                scores[k] = np.linalg.norm(wTc1.inverse().compose(wTc2).translation())

        # Check the validity of scores
        if sum(scores) <= 0.0:
            raise Exception(
                "Sum of scores cannot be zero (or smaller than zero)! It must a bug somewhere"
            )

        if self.sampling_method in [
            TriangulationParam.UNIFORM,
            TriangulationParam.BASELINE,
        ]:
            sample_indices = np.random.choice(
                len(scores), size=num_hypotheses, replace=False, p=scores / scores.sum()
            )

        if self.sampling_method == TriangulationParam.MAX_TO_MIN:
            sample_indices = np.argsort(scores)[-num_hypotheses:]

        return sample_indices.tolist()

    def compute_track_reprojection_errors(
        self, triangulated_pt: Point3, track: SfmTrack2d
    ) -> List[float]:
        """
        Calculate all individual reprojection errors in a given track

        Args:
            triangulated point: triangulated 3D point
            track: the measurements of a track

        Returns:
            reprojection errors
        """
        errors = []
        for (i, uv) in track.measurements:
            camera = self.track_camera_list.get(i)
            # Project to camera
            uv_hat = camera.project(triangulated_pt)
            # Projection error in camera
            errors.append(np.linalg.norm(uv - uv_hat))
        return errors

    def extract_measurements(
        self, track: SfmTrack2d, inliers: List[bool]
    ) -> Tuple[CameraSetCal3Bundler, Point2Vector]:
        """
        Extract measurements in a track for triangulation.

        Args:
            track: feature track from which measurements are to be extracted
            inliers: a boolean list that indicates the validity of each measurements

        Returns:
            camera_track: Vector of individual camera calibrations
                  TODO: is it really this? for first and last measurement
            measurement_track: Vector of 2d points
                  TODO: is it really this? Observations corresponding to first and last measurements
        """
        track_cameras = CameraSetCal3Bundler()
        track_measurements = Point2Vector()  # vector of 2d points

        for (measurement, is_inlier) in zip(track.measurements, inliers):
            if is_inlier:
                i, uv = measurement  # pull out camera index i and uv
                track_cameras.append(self.track_camera_list.get(i))
                track_measurements.append(uv)

        if len(track_cameras) < 2 or len(track_measurements) < 2:
            raise Exception(
                "Nb of measurements should not be <= 2. \
                    number of cameras is: {} \
                    and number of observations is {}".format(
                    len(track_cameras), len(track_measurements)
                )
            )

        return track_cameras, track_measurements

    def create_track_from_inliers(
        self, triangulated_pt: Point3, track: SfmTrack2d, inlier: List[bool]
    ) -> gtsam.SfmTrack:
        """
        Generate track based on inliers

        Args:
            triangulated_pt: triangulated 3d point
            track: list of 2d measurements each of the form (i,uv)
            inlier: best inlier list from RANSAC or all points

        Returns:
            SfmTrack object
        """
        # we will create a new track with only the inlier measurements
        new_track = gtsam.SfmTrack(triangulated_pt)

        for (i, uv) in track.measurements:
            if inlier[i]:
                new_track.add_measurement(i, uv)
        return new_track
