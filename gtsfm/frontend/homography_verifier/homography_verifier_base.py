"""Base class for the V (verification) stage of the frontend.

Authors: Ayush Baid, John Lambert
"""
import abc
from typing import Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Rot3, Unit3

from gtsfm.common.keypoints import Keypoints


MIN_PTS_HOMOGRAPHY = 4


class HomographyVerifierBase(metaclass=abc.ABCMeta):
    """Base class for all verifiers.

    Verifiers take the coordinates of the matches as inputs and returns the estimated essential matrix as well as
    geometrically verified points.
    """

    @abc.abstractmethod
    def verify(
        self, keypoints_i1: Keypoints, keypoints_i2: Keypoints, match_indices: np.ndarray, estimation_threshold_px: float
    ) -> Tuple[np.ndarray, float, int, np.ndarray]:
        """Verify that a set of correspondences belong to a homography configuration.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2).
            estimation_threshold_px: threshold value (in pixels) to use for classifying inliers in RANSAC.

        Returns:
            H: array of shape (3,3) representing homography matrix.
            inlier_idxs: indices of inliers from matches array.
            inlier_ratio: i.e. ratio of correspondences which approximately agree with planar geometry.
            num_inliers: number of correspondence consistent with estimated homography H.
        """

    def create_computation_graph(
        self,
        keypoints_i1_graph: Delayed,
        keypoints_i2_graph: Delayed,
        matches_i1i2_graph: Delayed,
        intrinsics_i1_graph: Delayed,
        intrinsics_i2_graph: Delayed,
    ) -> Tuple[Delayed, Delayed, Delayed]:
        """Generates the computation graph to perform verification of putative correspondences.

        Args:
            image_pair_indices: 2-tuple (i1,i2) specifying image pair indices
            detection_graph: nodes with features for each image.
            matcher_graph: nodes with matching results for pairs of images.
            camera_intrinsics_graph: nodes with intrinsics for each image.

        Returns:
            Delayed dask task for rotation i2Ri1 for specific image pair.
            Delayed dask task for unit translation i2Ui1 for specific image pair.
            Delayed dask task for indices of verified correspondence indices for the specific image pair.
            Delayed dask task for inlier ratio w.r.t. the estimated model, i.e. #final RANSAC inliers/ #putatives.
        """
        # we cannot immediately unpack the result tuple, per dask syntax
        result = dask.delayed(self.verify)(
            keypoints_i1_graph, keypoints_i2_graph, matches_i1i2_graph, intrinsics_i1_graph, intrinsics_i2_graph
        )
        i2Ri1_graph = result[0]
        i2Ui1_graph = result[1]
        v_corr_idxs_graph = result[2]
        inlier_ratio_est_model = result[3]

        return i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph, inlier_ratio_est_model
