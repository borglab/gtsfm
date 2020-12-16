""" Create 2D-3D data association as a precursor to Bundle Adjustment.
1. Forms feature tracks from verified correspondences and global poses.
2. Triangulates 3D world points for each track (Ransac and simple triangulation modes available)
3. Filters tracks based on reprojection error.

References: 
1. Richard I. Hartley and Peter Sturm. Triangulation. Computer Vision and Image Understanding, Vol. 68, No. 2, November, pp. 146–157, 1997
2. P. Moulon, P. Monasse. Unordered Feature Tracking Made Fast and Easy, 2012, HAL Archives.

Authors: Sushmita Warrier, Xiaolong Wu
"""

from typing import Dict, List, Tuple, Optional

import dask
from dask.delayed import Delayed
import gtsam
import numpy as np

from common.keypoints import Keypoints
from data_association.feature_tracks import FeatureTrackGenerator
from enum import Enum
from gtsam import (
    CameraSetCal3Bundler,
    PinholeCameraCal3Bundler,
    Point3,
    Point2Vector,
    triangulatePoint3,
)


class TriangulationParam(Enum):
    UNIFORM = 1
    BASELINE = 2
    MAX_TO_MIN = 3


class DataAssociation:
    """Class to form feature tracks; for each track, call LandmarkInitializer."""

    def __init__(self, reproj_error_thresh: float, min_track_len: int) -> None:
        """Initializes the hyperparameters.

        Args:
            reproj_error_thresh: the maximum reprojection error allowed.
            min_track_len: min length required for valid feature track / min nb of
                supporting views required for a landmark to be valid
        """
        self.reproj_error_thresh = reproj_error_thresh
        self.min_track_len = min_track_len

    def run(
        self,
        corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
        use_ransac: bool,
        cameras: Dict[int, PinholeCameraCal3Bundler],
        sampling_method: Optional[TriangulationParam] = None,
        num_samples: Optional[int] = None,
    ) -> gtsam.SfmData:
        """Perform data association

        Args:
            corr_idxs_dict: dictionary, with key as image pair (i1,i2) and value
                            as matching keypoint indices.
            keypoints_list: keypoints for each image.
            use_ransac: flag to set the usage of ransac-based triangulation or not
            cameras: dictionary with image index as key, and camera object w/
                     intrinsics + extrinsics as value.
            sampling_method: sampling method for ransac-based triangulation
            num_samples: number of samples in ransac-based triangulation
        """
        triangulated_landmark_map = gtsam.SfmData()
        tracks = FeatureTrackGenerator(corr_idxs_dict, keypoints_list)
        sfmdata_landmark_map = tracks.filtered_landmark_data

        # point indices are represented as j
        # nb of 3D points = nb of tracks, hence track_idx represented as j
        LMI = LandmarkInitializer(cameras)

        for j in range(len(sfmdata_landmark_map)):
            filtered_track = LMI.triangulate(
                sfmdata_landmark_map[j],
                use_ransac,
                sampling_method,
                num_samples,
                self.reproj_error_thresh,
            )

            if filtered_track.number_measurements() >= self.min_track_len:
                triangulated_landmark_map.add_track(filtered_track)
            else:
                print(
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
        use_ransac: bool,
        cameras: Dict[int, PinholeCameraCal3Bundler],
        sampling_method: Optional[TriangulationParam] = None,
        num_samples: Optional[int] = None,
    ) -> Delayed:
        """Creates a computation graph for performing data association.

        Args:
            corr_idxs_graph: dictionary of correspondence indices, each value
                             wrapped up as Delayed.
            cameras: list of cameras wrapped up as Delayed.
            use_ransac: whether or not to use ransac

            keypoints_graph: list of wrapped up keypoints for each image

        Returns:
            SfmData
        """

        return dask.delayed(self.run)(
            corr_idxs_dict,
            keypoints_list,
            use_ransac,
            cameras,
            sampling_method,
            num_samples,
        )


class LandmarkInitializer:
    """
    Class to initialize landmark points via triangulation w or w/o RANSAC inlier/outlier selection
    """

    def __init__(self, track_cameras: Dict[int, PinholeCameraCal3Bundler]) -> None:
        """
        Args:
            track_cameras: List of cameras
        """
        self.track_camera_list = track_cameras

    def triangulate(
        self,
        track: List,
        use_ransac: bool,
        sampling_method: Optional[TriangulationParam] = None,
        num_samples: Optional[int] = None,
        thresh: Optional[float] = None,
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
            if num_samples > len(matches):
                num_samples = len(matches)

            # Sampling
            samples = self.sampling(track, matches, sampling_method, num_samples)

            # Initialize the best output containers
            best_pt = Point3()
            best_votes = 0
            best_error = 1e10
            best_inliers = []

            for s in range(num_samples):
                k1, k2 = matches[samples[s]]

                idx1, pt1 = track[k1]
                idx2, pt2 = track[k2]

                camera_estimates = CameraSetCal3Bundler()
                camera_estimates.append(self.track_camera_list.get(idx1))
                camera_estimates.append(self.track_camera_list.get(idx2))

                img_measurements = Point2Vector()
                img_measurements.append(pt1)
                img_measurements.append(pt2)

                # triangulate point for track
                triangulated_pt = triangulatePoint3(
                    camera_estimates, img_measurements, rank_tol=1e-9, optimize=True
                )

                errors = self.reprojection_error(triangulated_pt, track)
                # TODO: Xiaolong - add inline comments to explain logic
                votes = [err < thresh for err in errors]

                sum_error = sum(errors)
                sum_votes = sum([int(v) for v in votes])

                if (sum_votes > best_votes) or (
                    sum_votes == best_votes and best_error > sum_error
                ):
                    best_votes = sum_votes
                    best_error = sum_error
                    best_pt = triangulated_pt
                    best_inliers = votes
        else:
            best_inliers = [True for k in range(len(track))]

        camera_track, measurement_track = self.extract_measurements(track, best_inliers)

        triangulated_track = dict()
        triangulated_pt_track = triangulatePoint3(
            camera_track, measurement_track, rank_tol=1e-9, optimize=True
        )
        triangulated_track.update({tuple(triangulated_pt_track): track})

        # we may want to compare the initialized best_pt with triangulated_pt_track
        return self.inlier_to_track(triangulated_track, best_inliers)

    def generate_matches(self, track: List) -> List[Tuple[int, int]]:
        """
        Extract all possible measurement pairs (k1, k2) in a track for triangulation.

        Args:
            track: feature track from which measurements are to be extracted
        Returns:
            matches: all possible matches in a given track
        """
        matches = []

        for k1 in range(len(track)):
            for k2 in range(k1 + 1, len(track)):
                matches.append([k1, k2])

        return matches

    def sampling(
        self,
        track: List,
        matches: List,
        sampling_method: TriangulationParam,
        num_samples: int,
    ) -> List[int]:
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
        # Initialize scores as uniform distribution
        scores = np.ones(len(matches), dtype=float)

        if sampling_method in [
            TriangulationParam.BASELINE,
            TriangulationParam.MAX_TO_MIN,
        ]:
            for k in range(len(matches)):
                k1, k2 = matches[k]

                idx1, pt1 = track[k1]
                idx2, pt2 = track[k2]

                wTc1 = self.track_camera_list.get(idx1).pose()
                wTc2 = self.track_camera_list.get(idx2).pose()

                # it is not a very correct approximation of depth, will do it better later
                scores[k] = np.linalg.norm(wTc1.compose(wTc2.inverse()).translation())

        # Check the validity of scores
        if sum(scores) <= 0.0:
            raise Exception(
                "Sum of scores cannot be zero (or smaller than zero)! It must a bug somewhere"
            )

        if sampling_method in [TriangulationParam.UNIFORM, TriangulationParam.BASELINE]:
            sample_index = np.random.choice(
                len(scores), num_samples, replace=False, p=scores / scores.sum()
            )

        if sampling_method == TriangulationParam.MAX_TO_MIN:
            sample_index = np.argsort(scores)[-num_samples:]

        return sample_index.tolist()

    def reprojection_error(self, triangulated_pt: Point3, track: List) -> List[float]:
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
            camera = self.track_camera_list.get(i)
            # Project to camera
            uv = camera.project(triangulated_pt)
            # Projection error in camera
            errors.append(np.linalg.norm(measurement - uv))
        return errors

    def extract_measurements(
        self, track: List, inliers: List
    ) -> Tuple[List, Point2Vector]:
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

        camera_track = CameraSetCal3Bundler()
        measurement_track = Point2Vector()

        for k in range(len(track)):
            if inliers[k]:
                img_idx, img_Pt = track[k]
                camera_track.append(self.track_camera_list.get(img_idx))
                measurement_track.append(img_Pt)

        if len(camera_track) < 2 or len(measurement_track) < 2:
            raise Exception(
                "Nb of measurements should not be <= 2. \
                    number of cameras is: {} \
                    and number of observations is {}".format(
                    len(camera_track), len(measurement_track)
                )
            )

        return camera_track, measurement_track

    def inlier_to_track(self, triangulated_track: Dict, inlier: List) -> gtsam.SfmTrack:
        """
        Generate track based on inliers

        Args:
            triangulated_track: with triangulated pt as key and track as value
            inlier: best inlier list from ransac or all points

        Returns:
            SfmTrack object
        """
        new_track = gtsam.SfmTrack(list(triangulated_track.keys())[0])

        for triangulated_pt, track in triangulated_track.items():
            for (i, measurement) in track:
                if inlier[i]:
                    new_track.add_measurement(i, measurement)
        return new_track
