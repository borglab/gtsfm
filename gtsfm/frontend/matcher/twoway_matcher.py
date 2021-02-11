"""Two way (mutual nearest neighbor) matcher.

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np

from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.matcher.matcher_base import MatchingDistanceType


class TwoWayMatcher(MatcherBase):
    """Two way (mutual nearest neighbor) matcher using OpenCV."""

    def match(
        self,
        descriptors_im1: np.ndarray,
        descriptors_im2: np.ndarray,
        distance_type: MatchingDistanceType = MatchingDistanceType.EUCLIDEAN,
    ) -> np.ndarray:
        """Match descriptor vectors.

        Output format:
        1. Each row represents a match.
        2. First column represents descriptor index from image #1.
        3. Second column represents descriptor index from image #2.
        4. Matches are sorted in descending order of the confidence (score).

        Args:
            descriptors_im1: descriptors from image #1, of shape (N1, D).
            descriptors_im2: descriptors from image #2, of shape (N2, D).
            distance_type (optional): the space to compute the distance between descriptors. Defaults to
                                      MatchingDistanceType.EUCLIDEAN.

        Returns:
            Match indices (sorted by confidence), as matrix of shape (N, 2), where N < min(N1, N2).
        """
        if distance_type is MatchingDistanceType.EUCLIDEAN:
            distance_metric = cv.NORM_L2
        elif distance_type is MatchingDistanceType.HAMMING:
            distance_metric = cv.NORM_HAMMING
        else:
            raise NotImplementedError("The distance type is not in MatchingDistanceType")

        if descriptors_im1.size == 0 or descriptors_im2.size == 0:
            return np.array([])

        # we will have to remove NaNs by ourselves
        valid_idx_im1 = np.nonzero(~(np.isnan(descriptors_im1).any(axis=1)))[0]
        valid_idx_im2 = np.nonzero(~(np.isnan(descriptors_im2).any(axis=1)))[0]

        descriptors_1 = descriptors_im1[valid_idx_im1]
        descriptors_2 = descriptors_im2[valid_idx_im2]

        # run OpenCV's matcher
        bf = cv.BFMatcher(normType=distance_metric, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda r: r.distance)

        match_indices = np.array([[m.queryIdx, m.trainIdx] for m in matches]).astype(np.int32)

        if match_indices.size == 0:
            return np.array([])

        # remap them back
        match_indices[:, 0] = valid_idx_im1[match_indices[:, 0]]
        match_indices[:, 1] = valid_idx_im2[match_indices[:, 1]]

        return match_indices
