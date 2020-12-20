""" Create 2D-3D data association as a precursor to Bundle Adjustment.
1. Forms feature tracks from verified correspondences and global poses.
2. Estimates 3D landmark for each track (Ransac and simple triangulation modes
   available)
3. Filters tracks based on reprojection error.

References: 
1. Richard I. Hartley and Peter Sturm. Triangulation. Computer Vision and Image
Understanding, Vol. 68, No. 2, November, pp. 146–157, 1997

Authors: Sushmita Warrier, Xiaolong Wu
"""
import itertools
import logging
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple

import dask
import gtsam
import numpy as np
from dask.delayed import Delayed
from gtsam import (
    CameraSetCal3Bundler,
    PinholeCameraCal3Bundler,
    Point2Vector,
    Point3,
    SfmData,
    SfmTrack,
)

import data_association.feature_tracks as feature_tracks
from common.keypoints import Keypoints

MAX_TRACK_REPROJ_ERROR = np.finfo(np.float32).max
SVD_DLT_RANK_TOL = 1e-9
NUM_SAMPLES_PER_RANSAC_HYPOTHESIS = 2

"""We have different modes for robust and non-robust triangulation.
In case of noise-free measurements, all the entries in a track are used w/o ransac.
If one of the three sampling modes for robust triangulation is selected, a pair
of cameras will be sampled."""


class TriangulationParam(Enum):
    NO_RANSAC = 0  # do not use filtering
    RANSAC_SAMPLE_UNIFORM = 1  # sample a pair of cameras uniformly at random
    RANSAC_SAMPLE_BIASED_BASELINE = (
        2  # sample pair of cameras based on largest estimated baseline
    )
    RANSAC_TOPK_BASELINES = (
        3  # deterministically choose hypotheses with largest estimate baseline
    )


class DataAssociation(NamedTuple):
    """Class to form feature tracks; for each track, call LandmarkInitializer.

    Args:
        reproj_error_thresh: the maximum reprojection error allowed.
        min_track_len: min length required for valid feature track / min nb of
            supporting views required for a landmark to be valid.
        mode: triangulation mode, which dictates whether or not to use robust
              estimation.
        num_ransac_hypotheses (optional): number of hypothesis for RANSAC-based
              triangulation.
    """

    reproj_error_thresh: float
    min_track_len: int
    mode: TriangulationParam
    num_ransac_hypotheses: Optional[int] = None

    def run(
        self,
        cameras: Dict[int, PinholeCameraCal3Bundler],
        corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
    ) -> SfmData:
        """Perform the data association.

        Args:
            cameras: dictionary with image index as key, and camera object w/
                     intrinsics + extrinsics as value.
            corr_idxs_dict: dictionary, with key as image pair (i1,i2) and value
                            as matching keypoint indices.
            keypoints_list: keypoints for each image.

        Returns:
            cameras and tracks as SfmData
        """
        tracks = feature_tracks.generate_tracks(corr_idxs_dict, keypoints_list)

        point3d_initializer = Point3dInitializer(
            cameras,
            self.mode,
            self.num_ransac_hypotheses,
            self.reproj_error_thresh,
        )

        triangulated_data = SfmData()
        for track_2d in tracks:
            # triangulate and filter based on reprojection error
            sfm_track = point3d_initializer.triangulate(track_2d)

            if sfm_track is not None:
                if sfm_track.number_measurements() >= self.min_track_len:
                    triangulated_data.add_track(sfm_track)
                else:
                    logging.info(
                        "Track length {} < {} discarded".format(
                            sfm_track.number_measurements(),
                            self.min_track_len,
                        )
                    )

        # TODO: improve dropped camera handling
        num_cameras = len(cameras.keys())
        expected_camera_indices = np.arange(num_cameras)
        # add cameras to landmark_map
        for i, cam in enumerate(cameras.values()):
            if i != expected_camera_indices[i]:
                raise RuntimeError("Some cameras must have been dropped ")
            triangulated_data.add_camera(cam)

        return triangulated_data

    def create_computation_graph(
        self,
        cameras: Delayed,
        corr_idxs_graph: Dict[Tuple[int, int], Delayed],
        keypoints_graph: List[Delayed],
    ) -> Delayed:
        """Creates a computation graph for performing data association.

        Args:
            cameras: list of cameras wrapped up as Delayed.
            corr_idxs_graph: dictionary of correspondence indices, each value
                             wrapped up as Delayed.
            keypoints_graph: list of wrapped up keypoints for each image.

        Returns:
            SfmData object wrapped up using dask.delayed.
        """
        return dask.delayed(self.run)(cameras, corr_idxs_graph, keypoints_graph)


class Point3dInitializer(NamedTuple):
    """Class to initialize landmark points via triangulation w/ or w/o RANSAC
    inlier/outlier selection.

    Note: We currently limit the size of each sample to 2 camera views in our
    RANSAC scheme.

    Args:
        track_cameras: Dict of cameras and their indices.
        mode: triangulation mode, which dictates whether or not to use robust
              estimation.
        num_ransac_hypotheses (optional): desired number of RANSAC hypotheses.
        reproj_error_thresh (optional): threshold for RANSAC inlier filtering.
    """

    track_camera_dict: Dict[int, PinholeCameraCal3Bundler]
    mode: TriangulationParam
    num_ransac_hypotheses: Optional[int] = None
    reproj_error_thresh: Optional[float] = None

    def triangulate(
        self, track: feature_tracks.SfmTrack2d
    ) -> Optional[SfmTrack]:
        """
        Triangulation based on selected triangulation mode, with resultinf tracks filtered based on reprojection error.

        Args:
            track: feature track from which measurements are to be extracted

        Returns:
            SfmTrack with 3d point j and 2d measurements in multiple cameras i.
        """
        if self.mode in [
            TriangulationParam.RANSAC_SAMPLE_UNIFORM,
            TriangulationParam.RANSAC_SAMPLE_BIASED_BASELINE,
            TriangulationParam.RANSAC_TOPK_BASELINES,
        ]:
            # Generate all possible matches
            measurement_pairs = self.generate_measurement_pairs(track)

            # We limit the number of samples to the number of actual available matches
            num_hypotheses = min(
                self.num_ransac_hypotheses, len(measurement_pairs)
            )

            # Sampling
            samples = self.sample_ransac_hypotheses(
                track, measurement_pairs, num_hypotheses
            )

            # Initialize the best output containers
            best_votes = 0
            best_error = MAX_TRACK_REPROJ_ERROR
            best_inliers = [False] * len(track.measurements)

            for sample_idxs in samples:
                k1, k2 = measurement_pairs[sample_idxs]

                i1, pt1 = track.measurements[k1]
                i2, pt2 = track.measurements[k2]

                camera_estimates = CameraSetCal3Bundler()
                # check for unestimated cameras
                if (
                    self.track_camera_dict.get(i1) != None
                    and self.track_camera_dict.get(i2) != None
                ):
                    camera_estimates.append(self.track_camera_dict.get(i1))
                    camera_estimates.append(self.track_camera_dict.get(i2))

                    img_measurements = Point2Vector()
                    img_measurements.append(pt1)
                    img_measurements.append(pt2)

                    # triangulate point for track
                    try:
                        triangulated_pt = gtsam.triangulatePoint3(
                            camera_estimates,
                            img_measurements,
                            rank_tol=SVD_DLT_RANK_TOL,
                            optimize=True,
                        )
                    except RuntimeError as e:
                        # TODO: handle cheirality exception properly?
                        logging.error("Error while triangulating")
                        continue

                    errors = np.array(
                        self.compute_track_reprojection_errors(
                            triangulated_pt, track
                        )
                    )
                    # The best solution should correspond to the one with most inliers
                    # If the inlier number are the same, check the average error of inliers
                    votes = (errors < self.reproj_error_thresh).tolist()

                    inlier_errors = errors[votes]

                    if inlier_errors.size:

                        avg_error = (
                            np.array(inlier_errors).sum()
                            / inlier_errors.shape[0]
                        )

                        sum_votes = np.array(votes).astype(int).sum()

                        if (sum_votes > best_votes) or (
                            sum_votes == best_votes and avg_error < best_error
                        ):
                            best_votes = sum_votes
                            best_error = avg_error
                            best_inliers = votes
                else:
                    logging.warning(
                        "Unestimated cameras found at indices {} or {}. Skipping them.".format(
                            i1, i2
                        )
                    )

        elif self.mode == TriangulationParam.NO_RANSAC:
            best_inliers = [True] * len(track.measurements)

        inlier_idxs = (np.where(best_inliers)[0]).tolist()

        if len(inlier_idxs) < 2:
            return None

        camera_track, measurement_track = self.extract_measurements(
            track, inlier_idxs
        )

        triangulated_pt = gtsam.triangulatePoint3(
            camera_track,
            measurement_track,
            rank_tol=SVD_DLT_RANK_TOL,
            optimize=True,
        )

        # we may want to compare the initialized best_pt with triangulated_pt_track
        return self.create_track_from_inliers(
            triangulated_pt, track, inlier_idxs
        )

    def generate_measurement_pairs(
        self, track: feature_tracks.SfmTrack2d
    ) -> List[Tuple[int, int]]:
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
        track: feature_tracks.SfmTrack2d,
        measurement_pairs: List[Tuple[int, int]],
        num_hypotheses: int,
    ) -> List[int]:
        """Sample a list of hypotheses (camera pairs) to use during triangulation

        Args:
            track: feature track from which measurements are to be extracted
            measurement_pairs: all possible indices of pairs of measurements in a given track
            num_hypotheses: desired number of samples
        Returns:
            indexes of matches: index of selected match
        """
        # Initialize scores as uniform distribution
        scores = np.ones(len(measurement_pairs), dtype=float)

        if self.mode in [
            TriangulationParam.RANSAC_SAMPLE_BIASED_BASELINE,
            TriangulationParam.RANSAC_TOPK_BASELINES,
        ]:
            for k, (k1, k2) in enumerate(measurement_pairs):
                i1, pt1 = track.measurements[k1]
                i2, pt2 = track.measurements[k2]

                wTc1 = self.track_camera_dict.get(i1).pose()
                wTc2 = self.track_camera_dict.get(i2).pose()

                # rough approximation approximation of baseline between the 2 cameras
                scores[k] = np.linalg.norm(
                    wTc1.inverse().compose(wTc2).translation()
                )

        # Check the validity of scores
        if sum(scores) <= 0.0:
            raise Exception(
                "Sum of scores cannot be zero (or smaller than zero)! It must a bug somewhere"
            )

        if self.mode in [
            TriangulationParam.RANSAC_SAMPLE_UNIFORM,
            TriangulationParam.RANSAC_SAMPLE_BIASED_BASELINE,
        ]:
            sample_indices = np.random.choice(
                len(scores),
                size=num_hypotheses,
                replace=False,
                p=scores / scores.sum(),
            )

        if self.mode == TriangulationParam.RANSAC_TOPK_BASELINES:
            sample_indices = np.argsort(scores)[-num_hypotheses:]

        return sample_indices.tolist()

    def compute_track_reprojection_errors(
        self, triangulated_pt: Point3, track: feature_tracks.SfmTrack2d
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
        for (i, uv_measured) in track.measurements:
            camera = self.track_camera_dict.get(i)
            # Project to camera
            uv = camera.project(triangulated_pt)
            # Projection error in camera
            errors.append(np.linalg.norm(uv_measured - uv))
        return errors

    def extract_measurements(
        self, track: feature_tracks.SfmTrack2d, inlier_idxs: List[int]
    ) -> Tuple[CameraSetCal3Bundler, Point2Vector]:
        """
        Extract measurements in a track for triangulation.

        Args:
            track: feature track from which measurements are to be extracted
            inliers: a boolean list that indicates the validity of each measurements

        Returns:
            track_cameras: Vector of individual camera calibrations pertaining to track
            track_measurements: Vector of 2d points pertaining to track measurements
        """
        track_cameras = CameraSetCal3Bundler()
        track_measurements = Point2Vector()  # vector of 2d points

        for idx in inlier_idxs:
            i, uv = track[idx]

            # check for unestimated cameras
            if self.track_camera_dict.get(i) != None:
                track_cameras.append(self.track_camera_dict.get(i))
                track_measurements.append(uv)
            else:
                logging.warning(
                    "Unestimated cameras found at index {}. Skipping them.".format(
                        i
                    )
                )

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
        self,
        triangulated_pt: Point3,
        track: feature_tracks.SfmTrack2d,
        inlier_idxs: List[int],
    ) -> SfmTrack:
        """
        Generate track based on inliers

        Args:
            triangulated_pt: triangulated 3d point
            track: list of 2d measurements each of the form (i,uv)
            inliers: best inlier list from RANSAC or all points

        Returns:
            SfmTrack object
        """
        # we will create a new track with only the inlier measurements
        new_track = SfmTrack(triangulated_pt)

        for idx in inlier_idxs:
            i, uv = track[idx]
            new_track.add_measurement(i, uv)

        return new_track
