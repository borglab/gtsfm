"""Algorithms to initialize 3D landmark point from measurements at known camera poses.

References:
1. Richard I. Hartley and Peter Sturm. Triangulation. Computer Vision and Image Understanding, Vol. 68, No. 2,
   November, pp. 146–157, 1997

Authors: Sushmita Warrier, Xiaolong Wu, John Lambert, Travis Driver
"""

import itertools
import sys
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple

import gtsam
import numpy as np

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.reprojection as reproj_utils
import gtsfm.utils.tracks as track_utils
from gtsfm.common.sfm_track import SfmTrack2d

NUM_SAMPLES_PER_RANSAC_HYPOTHESIS = 2
SVD_DLT_RANK_TOL = 1e-9
MAX_TRACK_REPROJ_ERROR = np.finfo(np.float32).max

logger = logger_utils.get_logger()

"""We have different modes for robust and non-robust triangulation. In case of noise-free measurements, all the entries
in a track are used w/o ransac. If one of the three sampling modes for robust triangulation is selected, a pair of
cameras will be sampled."""


class TriangulationExitCode(Enum):
    """Exit codes for triangulation computation."""

    # TODO(travisdriver): enforce exit codes in unit tests
    SUCCESS = 0  # successfully estimated 3d point from measurements
    CHEIRALITY_FAILURE = 1  # cheirality exception from gtsam.triangulatePoint3
    INLIERS_UNDERCONSTRAINED = 2  # insufficent number of inlier measurements
    POSES_UNDERCONSTRAINED = 3  # insufficent number of estimated camera poses
    EXCEEDS_REPROJ_THRESH = 4  # estimated 3d point exceeds reprojection threshold
    LOW_TRIANGULATION_ANGLE = 5  # maximum triangulation angle lower than threshold


class TriangulationSamplingMode(str, Enum):
    """Triangulation modes.

    NO_RANSAC: do not use filtering.
    RANSAC_SAMPLE_UNIFORM: sample a pair of cameras uniformly at random.
    RANSAC_SAMPLE_BIASED_BASELINE: sample pair of cameras based by largest estimated baseline.
    RANSAC_TOPK_BASELINES: deterministically choose hypotheses with largest estimate baseline.
    """

    NO_RANSAC = "NO_RANSAC"
    RANSAC_SAMPLE_UNIFORM = "RANSAC_SAMPLE_UNIFORM"
    RANSAC_SAMPLE_BIASED_BASELINE = "RANSAC_SAMPLE_BIASED_BASELINE"
    RANSAC_TOPK_BASELINES = "RANSAC_TOPK_BASELINES"


class TriangulationOptions(NamedTuple):
    """Options for triangulation solver.

    Based upon COLMAP's RANSAC class:
    Reference: https://github.com/colmap/colmap/blob/dev/src/optim/ransac.h

    See the following slides for a derivation of the #req'd samples: http://www.cse.psu.edu/~rtc12/CSE486/lecture15.pdf

    Args:
        mode: Triangulation mode, which dictates whether or not to use robust estimation.
        reproj_error_threshold: The maximum reprojection error allowed.
        min_triangulation_angle: Threshold for the minimum angle (in degrees) subtended at the triangulated track from 2
            cameras. Tracks for which the maximum angle at any two cameras is less then this threshold will be rejected.
        min_inlier_ratio: A priori assumed minimum probability that a point is an inlier.
        confidence: Desired confidence that at least one hypothesis is outlier free.
        dyn_num_hypotheses_multiplier: multiplication factor for dynamically computed hyptheses based on confidence.
        min_num_hypotheses: Minimum number of hypotheses.
        max_num_hypotheses: Maximum number of hypotheses.
    """

    mode: TriangulationSamplingMode
    reproj_error_threshold: float = np.inf  # defaults to no filtering unless specified
    min_triangulation_angle: float = 0.0

    # RANSAC parameters
    min_inlier_ratio: float = 0.1
    confidence: float = 0.9999
    dyn_num_hypotheses_multiplier: float = 3.0
    min_num_hypotheses: int = 0
    max_num_hypotheses: int = sys.maxsize

    def __check_ransac_params(self) -> None:
        """Check that input value are valid"""
        assert self.reproj_error_threshold > 0
        assert 0 < self.min_inlier_ratio < 1
        assert 0 < self.confidence < 1
        assert self.dyn_num_hypotheses_multiplier > 0
        assert 0 <= self.min_num_hypotheses < self.max_num_hypotheses

    def num_ransac_hypotheses(self) -> int:
        """Compute maximum number of hypotheses.

        The RANSAC module defaults to 2749 iterations, computed as:
            np.log(1-0.9999) / np.log( 1 - 0.1 **2) * 3 = 2749.3
        """
        self.__check_ransac_params()
        dyn_num_hypotheses = int(
            (np.log(1 - self.confidence) / np.log(1 - self.min_inlier_ratio**NUM_SAMPLES_PER_RANSAC_HYPOTHESIS))
            * self.dyn_num_hypotheses_multiplier
        )
        num_hypotheses = max(min(self.max_num_hypotheses, dyn_num_hypotheses), self.min_num_hypotheses)
        return num_hypotheses


class Point3dInitializer:
    """Class to initialize landmark points via triangulation w/ or w/o RANSAC inlier/outlier selection.

    Note: We currently limit the size of each sample to 2 camera views in our RANSAC scheme.
    Comparable to OpenSfM's `TrackTriangulator` class:
        https://github.com/mapillary/OpenSfM/blob/master/opensfm/reconstruction.py#L755

    Args:
        track_cameras: Dict of cameras and their indices.
        mode: Triangulation mode, which dictates whether or not to use robust estimation.
        reproj_error_thresh: Threshold on reproj errors for inliers.
        num_ransac_hypotheses (optional): Desired number of RANSAC hypotheses.
    """

    def __init__(self, track_camera_dict: Dict[int, gtsfm_types.CAMERA_TYPE], options: TriangulationOptions) -> None:
        self.track_camera_dict = track_camera_dict
        self.options = options

        if len(track_camera_dict) == 0:
            raise ValueError("No camera positions were estimated, so triangulation is not feasible.")

        sample_camera = list(self.track_camera_dict.values())[0]
        self._camera_set_class = gtsfm_types.get_camera_set_class_for_calibration(sample_camera.calibration())

    def execute_ransac_variant(self, track_2d: SfmTrack2d) -> np.ndarray:
        """Execute RANSAC algorithm to find best subset 2d measurements for a 3d point.
        RANSAC chooses one of 3 different sampling schemes to execute.

        Args:
            track: Feature track with N 2d measurements in separate images

        Returns:
            best_inliers: Boolean array of length N. Indices of measurements
               are set to true if they correspond to the best RANSAC hypothesis.
        """

        # Generate all possible matches.
        measurement_pairs = generate_measurement_pairs(track_2d)

        # Limit the number of samples to the number of available pairs.
        num_hypotheses = min(self.options.num_ransac_hypotheses(), len(measurement_pairs))

        # Sampling.
        samples = self.sample_ransac_hypotheses(track_2d, measurement_pairs, num_hypotheses)

        # Initialize the best output containers.
        best_num_votes = 0
        best_error = MAX_TRACK_REPROJ_ERROR
        best_inliers = np.zeros(len(track_2d.measurements), dtype=bool)
        for sample_idxs in samples:
            k1, k2 = measurement_pairs[sample_idxs]

            i1, uv1 = track_2d.measurements[k1]
            i2, uv2 = track_2d.measurements[k2]

            # Check for unestimated cameras.
            if self.track_camera_dict.get(i1) is None or self.track_camera_dict.get(i2) is None:
                logger.warning("Unestimated cameras found at indices %d or %d. Skipping them.", i1, i2)
                continue

            camera_estimates = self._camera_set_class()
            camera_estimates.append(self.track_camera_dict.get(i1))
            camera_estimates.append(self.track_camera_dict.get(i2))

            img_measurements = gtsam.Point2Vector()
            img_measurements.append(uv1)
            img_measurements.append(uv2)

            # Triangulate point for track.
            try:
                triangulated_pt = gtsam.triangulatePoint3(
                    camera_estimates,
                    img_measurements,
                    rank_tol=SVD_DLT_RANK_TOL,
                    optimize=True,
                )
            except RuntimeError:
                # TODO: handle cheirality exception properly?
                logger.debug(
                    "Cheirality exception from GTSAM's triangulatePoint3() likely due to outlier, skipping track"
                )
                continue

            errors, _ = reproj_utils.compute_point_reprojection_errors(
                self.track_camera_dict, triangulated_pt, track_2d.measurements
            )

            # The best solution should correspond to the one with most inliers
            # If the inlier number are the same, check the average error of inliers
            is_inlier = errors < self.options.reproj_error_threshold

            # Tally the number of votes.
            inlier_errors = errors[is_inlier]

            if inlier_errors.size > 0:
                # Only tally error over the inlier measurements.
                avg_error = inlier_errors.mean()
                num_votes = is_inlier.astype(int).sum()

                if (num_votes > best_num_votes) or (num_votes == best_num_votes and avg_error < best_error):
                    best_num_votes = num_votes
                    best_error = avg_error
                    best_inliers = is_inlier

        return best_inliers

    def triangulate(
        self, track_2d: SfmTrack2d
    ) -> Tuple[Optional[gtsam.SfmTrack], Optional[float], TriangulationExitCode]:
        """Triangulates 3D point according to the configured triangulation mode.

        Args:
            track: Feature track from which measurements are to be extracted.

        Returns:
            track with inlier measurements and 3D landmark. None returned if triangulation fails or has high error.
            avg_track_reproj_error: reprojection error of 3d triangulated point to each image plane
                Note: this may be "None" if the 3d point could not be triangulated successfully
                due to a cheirality exception or insufficient number of RANSAC inlier measurements
            is_cheirality_failure: boolean representing whether the selected 2d measurements lead
                to a cheirality exception upon triangulation
        """
        # Check if we will run RANSAC, or not.
        if self.options.mode in [
            TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM,
            TriangulationSamplingMode.RANSAC_SAMPLE_BIASED_BASELINE,
            TriangulationSamplingMode.RANSAC_TOPK_BASELINES,
        ]:
            best_inliers = self.execute_ransac_variant(track_2d)
        elif self.options.mode == TriangulationSamplingMode.NO_RANSAC:
            best_inliers = np.ones(len(track_2d.measurements), dtype=bool)  # all marked as inliers

        # Verify we have at least 2 inliers.
        inlier_idxs = (np.where(best_inliers)[0]).tolist()
        if len(inlier_idxs) < 2:
            return None, None, TriangulationExitCode.INLIERS_UNDERCONSTRAINED

        # Extract keypoint measurements corresponding to inlier indices.
        inlier_track = track_2d.select_subset(inlier_idxs)
        track_cameras, track_measurements = self.extract_measurements(inlier_track)

        # Exit if we do not have at least 2 measurements in cameras with estimated poses.
        if track_cameras is None:
            return None, None, TriangulationExitCode.POSES_UNDERCONSTRAINED

        # Triangulate and check for cheirality failure from GTSAM.
        try:
            triangulated_pt = gtsam.triangulatePoint3(
                track_cameras,
                track_measurements,
                rank_tol=SVD_DLT_RANK_TOL,
                optimize=True,
            )
        except RuntimeError:
            return None, None, TriangulationExitCode.CHEIRALITY_FAILURE

        # Compute reprojection errors for each measurement.
        reproj_errors, avg_track_reproj_error = reproj_utils.compute_point_reprojection_errors(
            self.track_camera_dict, triangulated_pt, inlier_track.measurements
        )

        # Check that all measurements are within reprojection error threshold.
        # TODO (travisdriver): Should we throw an error here if we're using RANSAC variant?
        if not np.all(reproj_errors.flatten() < self.options.reproj_error_threshold):
            return None, avg_track_reproj_error, TriangulationExitCode.EXCEEDS_REPROJ_THRESH

        # Create a gtsam.SfmTrack with the triangulated 3D point and associated 2D measurements.
        track_3d = gtsam.SfmTrack(triangulated_pt)
        for i, uv in inlier_track.measurements:
            track_3d.addMeasurement(i, uv)

        # Check that there is a sufficient triangulation angle.
        if (
            track_utils.get_max_triangulation_angle(track_3d, cameras=self.track_camera_dict)
            < self.options.min_triangulation_angle
        ):
            return None, avg_track_reproj_error, TriangulationExitCode.LOW_TRIANGULATION_ANGLE

        return track_3d, avg_track_reproj_error, TriangulationExitCode.SUCCESS

    def sample_ransac_hypotheses(
        self,
        track: SfmTrack2d,
        measurement_pairs: List[Tuple[int, ...]],
        num_hypotheses: int,
    ) -> List[int]:
        """Sample a list of hypotheses (camera pairs) to use during triangulation.

        Args:
            track: Feature track from which measurements are to be extracted.
            measurement_pairs: All possible indices of pairs of measurements in a given track.
            num_hypotheses: Desired number of samples.

        Returns:
            Indices of selected match.
        """
        # Initialize scores as uniform distribution
        scores = np.ones(len(measurement_pairs), dtype=float)

        if self.options.mode in [
            TriangulationSamplingMode.RANSAC_SAMPLE_BIASED_BASELINE,
            TriangulationSamplingMode.RANSAC_TOPK_BASELINES,
        ]:
            for k, (k1, k2) in enumerate(measurement_pairs):
                i1, _ = track.measurements[k1]
                i2, _ = track.measurements[k2]

                wTc1 = self.track_camera_dict[i1].pose()
                wTc2 = self.track_camera_dict[i2].pose()

                # Rough approximation approximation of baseline between the 2 cameras
                scores[k] = np.linalg.norm(wTc1.between(wTc2).translation())

        # Check the validity of scores.
        if sum(scores) <= 0.0:
            raise Exception("Sum of scores cannot be zero (or smaller than zero)! It must a bug somewhere")

        if self.options.mode in [
            TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM,
            TriangulationSamplingMode.RANSAC_SAMPLE_BIASED_BASELINE,
        ]:
            sample_indices = np.random.choice(
                len(scores),
                size=num_hypotheses,
                replace=False,
                p=scores / scores.sum(),
            )

        if self.options.mode == TriangulationSamplingMode.RANSAC_TOPK_BASELINES:
            sample_indices = np.argsort(scores)[-num_hypotheses:]

        return sample_indices.tolist()

    def extract_measurements(self, track: SfmTrack2d) -> Tuple[gtsfm_types.CAMERA_SET_TYPE, gtsam.Point2Vector]:
        """Convert measurements in a track into GTSAM primitive types for triangulation arguments.

        Returns None, None if less than 2 measurements were found with estimated camera poses after averaging.

        Args:
            track: Feature track from which measurements are to be extracted.

        Returns:
            Vector of individual camera calibrations pertaining to track
            Vector of 2d points pertaining to track measurements
        """
        track_cameras = self._camera_set_class()
        track_measurements = gtsam.Point2Vector()  # vector of 2d points

        # Compile valid measurements.
        for i, uv in track.measurements:
            # check for unestimated cameras
            if i in self.track_camera_dict and self.track_camera_dict.get(i) is not None:
                track_cameras.append(self.track_camera_dict.get(i))
                track_measurements.append(uv)
            else:
                logger.warning("Unestimated cameras found at index %d. Skipping them.", i)

        # Triangulation is underconstrained with <2 measurements.
        if len(track_cameras) < 2:
            return None, None

        return track_cameras, track_measurements


def generate_measurement_pairs(track: SfmTrack2d) -> List[Tuple[int, ...]]:
    """Extract all possible measurement pairs in a track for triangulation.

    Args:
        track: Feature track from which measurements are to be extracted.

    Returns:
        measurement_idxs: All possible matching measurement indices in a given track
    """
    num_track_measurements = track.number_measurements()
    all_measurement_idxs = range(num_track_measurements)
    measurement_pair_idxs = list(itertools.combinations(all_measurement_idxs, NUM_SAMPLES_PER_RANSAC_HYPOTHESIS))
    return measurement_pair_idxs
