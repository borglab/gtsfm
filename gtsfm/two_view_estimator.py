"""Estimator which operates on a pair of images to compute relative pose and verified indices.

Authors: Ayush Baid, John Lambert
"""
import logging
from typing import Tuple, Optional

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Pose3, Rot3, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils
import gtsfm.utils.serialization  # import needed to register serialization fns
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.verifier.verifier_base import VerifierBase

logger = logger_utils.get_logger()

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)


class TwoViewEstimator:
    """Wrapper for running two-view relative pose estimation on image pairs in the dataset."""

    def __init__(self, matcher: MatcherBase, verifier: VerifierBase, corr_metric_dist_threshold: int) -> None:
        """Initializes the two-view estimator from matcher and verifier.

        Args:
            matcher: matcher to use.
            verifier: verifier to use.
        """
        self.matcher = matcher
        self.verifier = verifier
        self._corr_metric_dist_threshold = corr_metric_dist_threshold

    def create_computation_graph(
        self,
        keypoints_i1_graph: Delayed,
        keypoints_i2_graph: Delayed,
        descriptors_i1_graph: Delayed,
        descriptors_i2_graph: Delayed,
        camera_intrinsics_i1_graph: Delayed,
        camera_intrinsics_i2_graph: Delayed,
        exact_intrinsics: bool = True,
        i2Ti1_expected_graph: Optional[Delayed] = None,
    ) -> Tuple[Delayed, Delayed, Delayed, Optional[Delayed], Optional[Delayed], Optional[Delayed]]:
        """Create delayed tasks for matching and verification.

        Args:
            keypoints_i1_graph: keypoints for image i1.
            keypoints_i2_graph: keypoints for image i2.
            descriptors_i1_graph: corr. descriptors for image i1.
            descriptors_i2_graph: corr. descriptors for image i2.
            camera_intrinsics_i1_graph: intrinsics for camera i1.
            camera_intrinsics_i2_graph: intrinsics for camera i2.
            exact_intrinsics (optional): flag to use intrinsics as exact. Defaults to True.
            i2Ti1_expected_graph (optional): ground truth relative pose, used for evaluation if available. Defaults to
                                             None.

        Returns:
            Computed relative rotation wrapped as Delayed.
            Computed relative translation direction wrapped as Delayed.
            Indices of verified correspondences wrapped as Delayed.
            Error in relative rotation wrapped as Delayed
            Error in relative translation direction wrapped as Delayed.
            Count of correct correspondences in output wrapped as Delayed.
        """

        # graph for matching to obtain putative correspondences
        corr_idxs_graph = self.matcher.create_computation_graph(descriptors_i1_graph, descriptors_i2_graph)

        # verification on putative correspondences to obtain relative pose
        # and verified correspondences
        (i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph,) = self.verifier.create_computation_graph(
            keypoints_i1_graph,
            keypoints_i2_graph,
            corr_idxs_graph,
            camera_intrinsics_i1_graph,
            camera_intrinsics_i2_graph,
            exact_intrinsics,
        )

        # if we have the expected data, evaluate the computed relative pose
        if i2Ti1_expected_graph is not None:
            pose_error_graphs = dask.delayed(compute_relative_pose_metrics)(
                i2Ri1_graph, i2Ui1_graph, i2Ti1_expected_graph
            )
            corr_error_graph = dask.delayed(compute_correspondence_metrics)(
                keypoints_i1_graph,
                keypoints_i2_graph,
                v_corr_idxs_graph,
                camera_intrinsics_i1_graph,
                camera_intrinsics_i2_graph,
                i2Ti1_expected_graph,
                self._corr_metric_dist_threshold,
            )
        else:
            pose_error_graphs = (None, None)
            corr_error_graph = None

        return (
            i2Ri1_graph,
            i2Ui1_graph,
            v_corr_idxs_graph,
            pose_error_graphs[0],
            pose_error_graphs[1],
            corr_error_graph,
        )


def compute_correspondence_metrics(
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    corr_idxs_i1i2: np.ndarray,
    intrinsics_i1: Cal3Bundler,
    intrinsics_i2: Cal3Bundler,
    i2Ti1: Pose3,
    epipolar_distance_threshold: float,
) -> Tuple[int, float]:
    """Compute the metrics for the generated verified correspondence.

    Args:
        keypoints_i1: detected keypoints in image i1.
        keypoints_i2: detected keypoints in image i2.
        corr_idxs_i1i2: indices of correspondences.
        intrinsics_i1: intrinsics for i1.
        intrinsics_i2: intrinsics for i2.
        i2Ti1: relative pose.
        epipolar_distance_threshold: max epipolar distance to qualify as a correct match.

    Returns:
        Number of correct correspondences.
        Ratio of correspondences which are correct.
    """
    number_correct = metric_utils.count_correct_correspondences(
        keypoints_i1.extract_indices(corr_idxs_i1i2[:, 0]),
        keypoints_i2.extract_indices(corr_idxs_i1i2[:, 1]),
        intrinsics_i1,
        intrinsics_i2,
        i2Ti1,
        epipolar_distance_threshold,
    )

    logger.debug(
        "[Two View Estimator] Correct Correspondences %d (ratio = %.2f)",
        number_correct,
        number_correct / corr_idxs_i1i2.shape[0],
    )

    return number_correct, number_correct / corr_idxs_i1i2.shape[0]


def compute_relative_pose_metrics(
    i2Ri1_computed: Optional[Rot3],
    i2Ui1_computed: Optional[Unit3],
    i2Ti1_expected: Pose3,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute the metrics on relative camera pose.

    Args:
        i2Ri1_computed: computed relative rotation.
        i2Ui1_computed: computed relative translation direction.
        i2Ti1_expected: expected relative pose.

    Returns:
        Rotation error, in degrees
        Unit translation error, in degrees
    """
    R_error_deg = comp_utils.compute_relative_rotation_angle(i2Ri1_computed, i2Ti1_expected.rotation())
    U_error_deg = comp_utils.compute_relative_unit_translation_angle(
        i2Ui1_computed, Unit3(i2Ti1_expected.translation())
    )

    return (R_error_deg, U_error_deg)
