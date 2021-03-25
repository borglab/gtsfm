"""Base class for the joint matcher-verifier stage of the frontend.

Authors: Ayush Baid
"""
import abc
from typing import Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Rot3, Unit3

from gtsfm.common.keypoints import Keypoints


class MatcherVerifierBase(metaclass=abc.ABCMeta):
    """Base class for all matcher-verifiers.

    MatcherVerifier takes the keypoints and descriptors of a pair of images, and produces relative pose, relative
    translations, and indices of verified correspondences.
    """

    @abc.abstractmethod
    def match_and_verify_with_exact_intrinsics(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Matches the descriptors to generate putative correspondences, and then verifies them to estimate the
        essential matrix and verified correspondences.

        Note: this function is preferred when camera intrinsics are known. The feature coordinates are normalized and
        the essential matrix is directly estimated.

        Args:
            keypoints_i1: detected features in image #i1, of length N1.
            keypoints_i2: detected features in image #i2, of length N2.
            descriptors_i1: descriptors for keypoints_i1.
            descriptors_i2: descriptors for keypoints_i2.
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= min(N1, N2).
        """

    @abc.abstractmethod
    def match_and_verify_with_approximate_intrinsics(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Matches the descriptors to generate putative correspondences, and then verifies them to estimate the
        fundamental matrix and verified correspondences.

        Note: this function is preferred when camera intrinsics are approximate (i.e from image size/exif). The feature
        coordinates are used to compute the fundamental matrix, which is then converted to the essential matrix.

        Args:
            keypoints_i1: detected features in image #i1, of length N1.
            keypoints_i2: detected features in image #i2, of length N2.
            descriptors_i1: descriptors for keypoints_i1.
            descriptors_i2: descriptors for keypoints_i2.
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= min(N1, N2).
        """

    def create_computation_graph(
        self,
        keypoints_i1_graph: Delayed,
        keypoints_i2_graph: Delayed,
        descriptors_i1_graph: Delayed,
        descriptors_i2_graph: Delayed,
        camera_intrinsics_i1_graph: Delayed,
        camera_intrinsics_i2_graph: Delayed,
        exact_intrinsics_flag: bool = True,
    ) -> Tuple[Delayed, Delayed, Delayed]:
        """Generates computation graph for recovering relative pose and verified correspondences from the keypoint and
        descriptor graph from 2 images.

        Args:
            keypoints_i1_graph: keypoints for image #i1, wrapped up in Delayed.
            keypoints_i2_graph: keypoints for image #i2, wrapped up in Delayed.
            descriptors_i1_graph: descriptors corr. to keypoints_i1_graph.
            descriptors_i2_graph (Delayed): descriptors corr. to keypoints_i2_graph.
            camera_intrinsics_i1_graph: intrinsics for image #i1, wrapped up in Delayed.
            camera_intrinsics_i2_graph: intrinsics for image #i2, wrapped up in Delayed.
            exact_intrinsics_flag (optional): flag denoting if intrinsics are exact, and an essential matrix can be
                                              directly computed. Defaults to True.

        Returns:
            Delayed dask task for rotation i2Ri1 for the image pair.
            Delayed dask task for unit translation i2Ui1 for the image pair.
            Delayed dask task for indices of verified correspondences.
        """

        fn_to_use = (
            self.match_and_verify_with_exact_intrinsics
            if exact_intrinsics_flag
            else self.match_and_verify_with_approximate_intrinsics
        )

        result = dask.delayed(fn_to_use)(
            keypoints_i1_graph,
            keypoints_i2_graph,
            descriptors_i1_graph,
            descriptors_i2_graph,
            camera_intrinsics_i1_graph,
            camera_intrinsics_i2_graph,
        )

        i2Ri1_graph = result[0]
        i2Ui1_graph = result[1]
        v_corr_idxs_graph = result[2]

        return i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph
