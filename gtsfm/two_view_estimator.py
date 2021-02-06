"""Estimator which operates on a pair of images to compute relative pose and
verified indices.

Authors: Ayush Baid, John Lambert
"""
import logging
import sys
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
from dask.delayed import Delayed
from gtsam import (
    PinholeCameraCal3Bundler,
    Pose3,
    Rot3,
    SfmData,
    Unit3,
)

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

    def __init__(self, matcher: MatcherBase, verifier: VerifierBase):
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
    ) -> Tuple[Delayed, Delayed, Delayed]:
        """Create delayed tasks for matching and verification."""

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

        return i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph