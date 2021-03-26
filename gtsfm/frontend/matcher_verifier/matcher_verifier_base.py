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
    def match_and_verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
        use_intrinsics_in_verification: bool = False,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Matches the descriptors to generate putative correspondences, and then verifies them to estimate the
        essential matrix and verified correspondences.

        Args:
            keypoints_i1: detected features in image #i1, of length N1.
            keypoints_i2: detected features in image #i2, of length N2.
            descriptors_i1: descriptors for keypoints_i1.
            descriptors_i2: descriptors for keypoints_i2.
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.
            use_intrinsics_in_verification (optional): Flag to perform keypoint normalization and compute the essential
                                                       matrix instead of fundamental matrix. This should be preferred
                                                       when the exact intrinsics are known as opposed to approximating
                                                       them from exif data. Defaults to False.

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
        use_intrinsics_in_verification: bool = False,
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
            use_intrinsics_in_verification (optional): Flag to perform keypoint normalization and compute the essential
                                                       matrix instead of fundamental matrix. This should be preferred
                                                       when the exact intrinsics are known as opposed to approximating
                                                       them from exif data. Defaults to False.

        Returns:
            Delayed dask task for rotation i2Ri1 for the image pair.
            Delayed dask task for unit translation i2Ui1 for the image pair.
            Delayed dask task for indices of verified correspondences.
        """
        result = dask.delayed(self.match_and_verify)(
            keypoints_i1_graph,
            keypoints_i2_graph,
            descriptors_i1_graph,
            descriptors_i2_graph,
            camera_intrinsics_i1_graph,
            camera_intrinsics_i2_graph,
            use_intrinsics_in_verification,
        )
        i2Ri1_graph = result[0]
        i2Ui1_graph = result[1]
        v_corr_idxs_graph = result[2]

        return i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph
