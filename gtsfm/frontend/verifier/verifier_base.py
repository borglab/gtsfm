"""Base class for the V (verification) stage of the frontend.

Authors: Ayush Baid
"""
import abc
from typing import Dict, List, Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Rot3, Unit3

from gtsfm.common.keypoints import Keypoints


class VerifierBase(metaclass=abc.ABCMeta):
    """Base class for all verifiers.

    Verifiers take the coordinates of the matches as inputs and returns the estimated essential matrix as well as
    geometrically verified points.
    """

    def __init__(self, min_pts):
        self.min_pts = min_pts

    @abc.abstractmethod
    def verify_with_exact_intrinsics(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is preferred when camera intrinsics are known. The feature coordinates are normalized and
        the essential matrix is directly estimated.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3. These are subset of match_indices.
        """

    @abc.abstractmethod
    def verify_with_approximate_intrinsics(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is preferred when camera intrinsics are approximate (i.e from image size/exif). The feature
        coordinates are used to compute the fundamental matrix, which is then converted to the essential matrix.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3. These are subset of match_indices.
        """

    def create_computation_graph(
        self,
        keypoints_i1_graph: Delayed,
        keypoints_i2_graph: Delayed,
        matches_i1i2_graph: Delayed,
        intrinsics_i1_graph: Delayed,
        intrinsics_i2_graph: Delayed,
        exact_intrinsics_flag: bool = True,
    ) -> Tuple[Delayed, Delayed, Delayed]:
        """Generates the computation graph to perform verification of putative
        correspondences.

        Args:
            image_pair_indices: 2-tuple (i1,i2) specifying image pair indices
            detection_graph: nodes with features for each image.
            matcher_graph: nodes with matching results for pairs of images.
            camera_intrinsics_graph: nodes with intrinsics for each image.
            exact_intrinsics_flag (optional): flag denoting if intrinsics are exact, and an essential matrix can be
                                              directly computed. Defaults to True.

        Returns:
            Delayed dask task for rotation i2Ri1 for specific image pair.
            Delayed dask task for unit translation i2Ui1 for specific image pair.
            Delayed dask task for indices of verified correspondence indices for the specific image pair
        """
        fn_to_use = (
            self.verify_with_exact_intrinsics if exact_intrinsics_flag else self.verify_with_approximate_intrinsics
        )

        # note that we cannot immediately unpack the result tuple, per dask syntax
        result = dask.delayed(fn_to_use)(
            keypoints_i1_graph,
            keypoints_i2_graph,
            matches_i1i2_graph,
            intrinsics_i1_graph,
            intrinsics_i2_graph,
        )
        i2Ri1_graph = result[0]
        i2Ui1_graph = result[1]
        v_corr_idxs_graph = result[2]

        return i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph
