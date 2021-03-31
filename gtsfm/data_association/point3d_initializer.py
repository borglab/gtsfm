"""Algorithms to initialize 3D landmark point from measurements at known camera poses.

References: 
1. Richard I. Hartley and Peter Sturm. Triangulation. Computer Vision and Image Understanding, Vol. 68, No. 2,
   November, pp. 146–157, 1997

Authors: Sushmita Warrier, Xiaolong Wu
"""

import itertools
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple

import gtsam
import numpy as np
from gtsam import (
    CameraSetCal3Bundler,
    PinholeCameraCal3Bundler,
    Point2Vector,
    SfmTrack,
)

import gtsfm.utils.logger as logger_utils
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.utils.reprojection import compute_point_reprojection_errors

NUM_SAMPLES_PER_RANSAC_HYPOTHESIS = 2
SVD_DLT_RANK_TOL = 1e-9
MAX_TRACK_REPROJ_ERROR = np.finfo(np.float32).max

logger = logger_utils.get_logger()

"""We have different modes for robust and non-robust triangulation. In case of noise-free measurements, all the entries
in a track are used w/o ransac. If one of the three sampling modes for robust triangulation is selected, a pair of
cameras will be sampled."""


class TriangulationParam(Enum):
    NO_RANSAC = 0  # do not use filtering
    RANSAC_SAMPLE_UNIFORM = 1  # sample a pair of cameras uniformly at random
    RANSAC_SAMPLE_BIASED_BASELINE = 2  # sample pair of cameras based on largest estimated baseline
    RANSAC_TOPK_BASELINES = 3  # deterministically choose hypotheses with largest estimate baseline


class Point3dInitializer(NamedTuple):
    """Class to initialize landmark points via triangulation w/ or w/o RANSAC inlier/outlier selection.

    Note: We currently limit the size of each sample to 2 camera views in our RANSAC scheme.

    Args:
        track_cameras: Dict of cameras and their indices.
        mode: triangulation mode, which dictates whether or not to use robust estimation.
        reproj_error_thresh: threshold on reproj errors for inliers.
        num_ransac_hypotheses (optional): desired number of RANSAC hypotheses.
    """

    track_camera_dict: Dict[int, PinholeCameraCal3Bundler]
    mode: TriangulationParam
    reproj_error_thresh: float
    num_ransac_hypotheses: Optional[int] = None

    def execute_ransac_variant(self, track_2d: SfmTrack2d) -> np.ndarray:
        """Execute RANSAC algorithm to find best subset 2d measurements for a 3d point.
        RANSAC chooses one of 3 different sampling schemes to execute.

        Args:
            track: feature track with N 2d measurements in separate images

        Returns:
            best_inliers: boolean array of length N. Indices of measurements
               are set to true if they correspond to the best RANSAC hypothesis
        """
        # Generate all possible matches
        measurement_pairs = self.generate_measurement_pairs(track_2d)

        # limit the number of samples to the number of available pairs
        num_hypotheses = min(self.num_ransac_hypotheses, len(measurement_pairs))

        # Sampling
        samples = self.sample_ransac_hypotheses(track_2d, measurement_pairs, num_hypotheses)

        # Initialize the best output containers
        best_num_votes = 0
        best_error = MAX_TRACK_REPROJ_ERROR
        best_inliers = np.zeros(len(track_2d.measurements), dtype=bool)

        for sample_idxs in samples:
            k1, k2 = measurement_pairs[sample_idxs]

            i1, uv1 = track_2d.measurements[k1]
            i2, uv2 = track_2d.measurements[k2]

            # check for unestimated cameras
            if self.track_camera_dict.get(i1) is None or self.track_camera_dict.get(i2) is None:
                logger.warning("Unestimated cameras found at indices {} or {}. Skipping them.".format(i1, i2))
                continue

            camera_estimates = CameraSetCal3Bundler()
            camera_estimates.append(self.track_camera_dict.get(i1))
            camera_estimates.append(self.track_camera_dict.get(i2))

            img_measurements = Point2Vector()
            img_measurements.append(uv1)
            img_measurements.append(uv2)

            # triangulate point for track
            try:
                triangulated_pt = gtsam.triangulatePoint3(
                    camera_estimates,
                    img_measurements,
                    rank_tol=SVD_DLT_RANK_TOL,
                    optimize=True,
                )
            except RuntimeError:
                # TODO: handle cheirality exception properly?
                logger.info(
                    "Cheirality exception from GTSAM's triangulatePoint3() likely due to outlier, skipping track"
                )
                continue

            errors, _ = compute_point_reprojection_errors(
                self.track_camera_dict, triangulated_pt, track_2d.measurements
            )

            # The best solution should correspond to the one with most inliers
            # If the inlier number are the same, check the average error of inliers
            is_inlier = errors < self.reproj_error_thresh

            # tally the number of votes
            inlier_errors = errors[is_inlier]

            if inlier_errors.size > 0:
                # only tally error over the inlier measurements
                avg_error = inlier_errors.mean()
                num_votes = is_inlier.astype(int).sum()

                if (num_votes > best_num_votes) or (num_votes == best_num_votes and avg_error < best_error):
                    best_num_votes = num_votes
                    best_error = avg_error
                    best_inliers = is_inlier

        return best_inliers

    def triangulate(self, track_2d: SfmTrack2d) -> Tuple[Optional[SfmTrack], Optional[float], bool]:
        """Triangulates 3D point according to the configured triangulation mode.

        Args:
            track: feature track from which measurements are to be extracted

        Returns:
            track with inlier measurements and 3D landmark. None returned if triangulation fails or has high error.
            avg_track_reproj_error: reprojection error of 3d triangulated point to each image plane
                Note: this may be "None" if the 3d point could not be triangulated successfully
                due to a cheirality exception or insufficient number of RANSAC inlier measurements
            is_cheirality_failure: boolean representing whether the selected 2d measurements lead
                to a cheirality exception upon triangulation
        """
        if self.mode in [
            TriangulationParam.RANSAC_SAMPLE_UNIFORM,
            TriangulationParam.RANSAC_SAMPLE_BIASED_BASELINE,
            TriangulationParam.RANSAC_TOPK_BASELINES,
        ]:
            best_inliers = self.execute_ransac_variant(track_2d)

        elif self.mode == TriangulationParam.NO_RANSAC:
            best_inliers = np.ones(len(track_2d.measurements), dtype=bool)  # all marked as inliers

        inlier_idxs = (np.where(best_inliers)[0]).tolist()

        is_cheirality_failure = False
        if len(inlier_idxs) < 2:
            return None, None, is_cheirality_failure

        inlier_track = track_2d.select_subset(inlier_idxs)

        camera_track, measurement_track = self.extract_measurements(inlier_track)
        try:
            triangulated_pt = gtsam.triangulatePoint3(
                camera_track,
                measurement_track,
                rank_tol=SVD_DLT_RANK_TOL,
                optimize=True,
            )
        except RuntimeError:
            is_cheirality_failure = True
            return None, None, is_cheirality_failure

        # compute reprojection errors for each measurement
        reproj_errors, avg_track_reproj_error = compute_point_reprojection_errors(
            self.track_camera_dict, triangulated_pt, inlier_track.measurements
        )

        # all the measurements should have error < threshold
        if not np.all(reproj_errors < self.reproj_error_thresh):
            return None, avg_track_reproj_error, is_cheirality_failure

        track_3d = SfmTrack(triangulated_pt)
        for i, uv in inlier_track.measurements:
            track_3d.add_measurement(i, uv)

        return track_3d, avg_track_reproj_error, is_cheirality_failure

    def generate_measurement_pairs(self, track: SfmTrack2d) -> List[Tuple[int, int]]:
        """
        Extract all possible measurement pairs in a track for triangulation.

        Args:
            track: feature track from which measurements are to be extracted

        Returns:
            measurement_idxs: all possible matching measurement indices in a given track
        """
        num_track_measurements = track.number_measurements()
        all_measurement_idxs = range(num_track_measurements)
        measurement_pair_idxs = list(itertools.combinations(all_measurement_idxs, NUM_SAMPLES_PER_RANSAC_HYPOTHESIS))
        return measurement_pair_idxs

    def sample_ransac_hypotheses(
        self,
        track: SfmTrack2d,
        measurement_pairs: List[Tuple[int, int]],
        num_hypotheses: int,
    ) -> List[int]:
        """Sample a list of hypotheses (camera pairs) to use during triangulation.

        Args:
            track: feature track from which measurements are to be extracted
            measurement_pairs: all possible indices of pairs of measurements in a given track
            num_hypotheses: desired number of samples
        Returns:
            Indices of selected match
        """
        # Initialize scores as uniform distribution
        scores = np.ones(len(measurement_pairs), dtype=float)

        if self.mode in [
            TriangulationParam.RANSAC_SAMPLE_BIASED_BASELINE,
            TriangulationParam.RANSAC_TOPK_BASELINES,
        ]:
            for k, (k1, k2) in enumerate(measurement_pairs):
                i1, _ = track.measurements[k1]
                i2, _ = track.measurements[k2]

                wTc1 = self.track_camera_dict[i1].pose()
                wTc2 = self.track_camera_dict[i2].pose()

                # rough approximation approximation of baseline between the 2 cameras
                scores[k] = np.linalg.norm(wTc1.inverse().compose(wTc2).translation())

        # Check the validity of scores
        if sum(scores) <= 0.0:
            raise Exception("Sum of scores cannot be zero (or smaller than zero)! It must a bug somewhere")

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

    def extract_measurements(self, track: SfmTrack2d) -> Tuple[CameraSetCal3Bundler, Point2Vector]:
        """Extract measurements in a track for triangulation.

        Args:
            track: feature track from which measurements are to be extracted.

        Returns:
            Vector of individual camera calibrations pertaining to track
            Vector of 2d points pertaining to track measurements
        """
        track_cameras = CameraSetCal3Bundler()
        track_measurements = Point2Vector()  # vector of 2d points

        for i, uv in track.measurements:

            # check for unestimated cameras
            if self.track_camera_dict.get(i) is not None:
                track_cameras.append(self.track_camera_dict.get(i))
                track_measurements.append(uv)
            else:
                logger.warning("Unestimated cameras found at index {}. Skipping them.".format(i))

        if len(track_cameras) < 2 or len(track_measurements) < 2:
            raise Exception(
                "Nb of measurements should not be <= 2. \
                    number of cameras is: {} \
                    and number of observations is {}".format(
                    len(track_cameras), len(track_measurements)
                )
            )

        return track_cameras, track_measurements
