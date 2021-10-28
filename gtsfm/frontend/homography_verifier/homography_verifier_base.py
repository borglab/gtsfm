"""Base class for the V (verification) stage of the frontend.

Authors: Ayush Baid, John Lambert
"""
import abc
from typing import Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed

from gtsfm.common.keypoints import Keypoints


MIN_PTS_HOMOGRAPHY = 4


class HomographyVerifierBase(metaclass=abc.ABCMeta):
    """Base class for all verifiers.

    Verifiers take the coordinates of the matches as inputs and returns the estimated essential matrix as well as
    geometrically verified points.
    """

    @abc.abstractmethod
    def verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        estimation_threshold_px: float,
    ) -> Tuple[Optional[np.ndarray], np.ndarray, float, int]:
        """Verify that a set of correspondences belong to a homography configuration.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2).
            estimation_threshold_px: threshold value (in pixels) to use for classifying inliers in RANSAC.

        Returns:
            H: array of shape (3,3) representing homography matrix.
            inlier_idxs: indices of inliers from matches array.
            inlier_ratio: i.e. ratio of correspondences which approximately agree with homography geometry
                (whether planar or panoramic).
            num_inliers: number of correspondences consistent with estimated homography H.
        """

    def create_computation_graph(
        self,
        keypoints_i1_graph: Delayed,
        keypoints_i2_graph: Delayed,
        matches_i1i2_graph: Delayed,
        estimation_threshold_px_graph: Delayed,
    ) -> Tuple[Delayed, Delayed, Delayed, Delayed]:
        """Generates the computation graph to perform verification of putative correspondences.

        Args:
            keypoints_i1_graph: keypoints for image #i1, wrapped in Delayed (evaluates to Keypoints).
            keypoints_i2_graph: keypoints for image #i2, wrapped in Delayed (evaluates to Keypoints).
            matches_i1i2_graph: indices of putative correspondences, wrapped in Delayed (evaluates to np.ndarray).
            estimation_threshold_px_graph: threshold value, wrapped in Delayed.

        Returns:
            Delayed dask task for homography i2Hi1 for specific image pair.
            Delayed dask task for indices of verified homography correspondence indices for the specific image pair.
            Delayed dask task for inlier ratio w.r.t. the estimated homography model,
                i.e. (#final RANSAC inliers)/ (#putatives).
            Delayed dask task for number of inliers w.r.t. the estimated homography model.
        """
        H_graph, H_inlier_idxs_graph, inlier_ratio_H_graph, num_inliers_H_graph = dask.delayed(self.verify, nout=4)(
            keypoints_i1_graph, keypoints_i2_graph, matches_i1i2_graph, estimation_threshold_px_graph
        )
        return H_graph, H_inlier_idxs_graph, inlier_ratio_H_graph, num_inliers_H_graph
