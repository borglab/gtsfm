"""Estimator which operates on a pair of images to compute relative pose and
verified indices.

Authors: Ayush Baid, John Lambert
"""
import logging
import sys
from typing import Tuple, Optional

import dask
from dask.delayed import Delayed
from gtsam import Pose3, Rot3, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.serialization  # import needed to register serialization fns
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.verifier.verifier_base import VerifierBase

# configure loggers to avoid DEBUG level stdout messages
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)


class TwoViewEstimator:
    """Wrapper for running two-view relative pose estimation on image pairs in
    the dataset."""

    def __init__(self, matcher: MatcherBase, verifier: VerifierBase) -> None:
        """Initializes the two-view estimator from matcher and verifier.

        Args:
            matcher: matcher to use.
            verifier: verifier to use.
        """
        self.matcher = matcher
        self.verifier = verifier

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
    ) -> Tuple[Delayed, Delayed, Delayed, Optional[Delayed], Optional[Delayed]]:
        """Create delayed tasks for matching and verification.

        Args:
            keypoints_i1_graph: keypoints for image i1.
            keypoints_i2_graph: keypoints for image i2.
            descriptors_i1_graph: corr. descriptors for image i1.
            descriptors_i2_graph: corr. descriptors for image i2.
            camera_intrinsics_i1_graph: intrinsics for camera i1.
            camera_intrinsics_i2_graph: intrinsics for camera i2.
            exact_intrinsics (optional): flag to use intrinsics as exact.
                                         Defaults to True.
            i2Ti1_expected_graph (optional): ground truth relative pose, used
                                             for evaluation if available.
                                             Defaults to None.

        Returns:
            Computed relative rotation wrapped as Delayed.
            Computed relative translation direction wrapped as Delayed.
            Indices of verified correspondences wrapped as Delayed.
            Error in relative rotation wrapped as Delayed
            Error in relative translation direction wrapped as Delayed.
        """

        # graph for matching to obtain putative correspondences
        corr_idxs_graph = self.matcher.create_computation_graph(
            descriptors_i1_graph, descriptors_i2_graph
        )

        # verification on putative correspondences to obtain relative pose
        # and verified correspondences
        (
            i2Ri1_graph,
            i2Ui1_graph,
            v_corr_idxs_graph,
        ) = self.verifier.create_computation_graph(
            keypoints_i1_graph,
            keypoints_i2_graph,
            corr_idxs_graph,
            camera_intrinsics_i1_graph,
            camera_intrinsics_i2_graph,
            exact_intrinsics,
        )

        # if we have the expected data, evaluate the computed relative pose
        if i2Ti1_expected_graph is not None:
            error_graphs = dask.delayed(compute_metrics)(
                i2Ri1_graph, i2Ui1_graph, i2Ti1_expected_graph
            )
        else:
            error_graphs = (None, None)

        return (
            i2Ri1_graph,
            i2Ui1_graph,
            v_corr_idxs_graph,
            error_graphs[0],
            error_graphs[1],
        )


def compute_metrics(
    i2Ri1_computed: Optional[Rot3],
    i2Ui1_computed: Optional[Unit3],
    i2Ti1_expected: Pose3,
) -> Tuple[float, float]:
    """Compute the metrics on relative camera pose.

    Args:
        i2Ri1_computed: computed relative rotation.
        i2Ui1_computed: computed relative translation direction.
        i2Ti1_expected: expected relative pose.

    Returns:
        Rotation error.
        Unit translation error.
    """

    R_error = comp_utils.compute_relative_rotation_angle(
        i2Ri1_computed, i2Ti1_expected.rotation()
    )

    U_error = comp_utils.compute_relative_unit_translation_angle(
        i2Ui1_computed, Unit3(i2Ti1_expected.translation())
    )

    logging.debug("[Two View Estimator] Relative rotation error %f", R_error)
    logging.debug(
        "[Two View Estimator] Relative unit-translation error %f", U_error
    )

    return (R_error, U_error)
