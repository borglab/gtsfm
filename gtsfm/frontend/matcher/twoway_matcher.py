"""Two way (mutual nearest neighbor) matcher.

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np
from enum import Enum

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.matcher_base import MatcherBase


class MatchingDistanceType(Enum):
    """Type of distance metric to use for matching descriptors."""

    HAMMING = 1
    EUCLIDEAN = 2


class TwoWayMatcher(MatcherBase):
    """Two way (mutual nearest neighbor) matcher using OpenCV."""

    def __init__(self, distance_type: MatchingDistanceType = MatchingDistanceType.EUCLIDEAN):
        super().__init__()
        self._distance_type = distance_type

    def match(
        self,
        keypoints_i1: Keypoints,  # pylint: disable=unused-argument
        keypoints_i2: Keypoints,  # pylint: disable=unused-argument
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
    ) -> np.ndarray:
        """Match descriptor vectors.

        Output format:
        1. Each row represents a match.
        2. First column represents keypoint index from image #i1.
        3. Second column represents keypoint index from image #i2.
        4. Matches are sorted in descending order of the confidence (score), if possible.

        Args:
            keypoints_i1: keypoints for image #i1, of length N1.
            keypoints_i2: keypoints for image #i2, of length N2.
            descriptors_i1: descriptors corr. to keypoints_i1.
            descriptors_i2: descriptors corr. to keypoints_i2.

        Returns:
            Match indices (sorted by confidence), as matrix of shape (N, 2), where N < min(N1, N2).
        """
        if self._distance_type is MatchingDistanceType.EUCLIDEAN:
            distance_metric = cv.NORM_L2
        elif self._distance_type is MatchingDistanceType.HAMMING:
            distance_metric = cv.NORM_HAMMING
        else:
            raise NotImplementedError("The distance type is not in MatchingDistanceType")

        if descriptors_i1.size == 0 or descriptors_i2.size == 0:
            return np.array([])

        # we will have to remove NaNs by ourselves
        valid_idx_i1 = np.nonzero(~(np.isnan(descriptors_i1).any(axis=1)))[0]
        valid_idx_i2 = np.nonzero(~(np.isnan(descriptors_i2).any(axis=1)))[0]

        descriptors_1 = descriptors_i1[valid_idx_i1]
        descriptors_2 = descriptors_i2[valid_idx_i2]

        # run OpenCV's matcher
        bf = cv.BFMatcher(normType=distance_metric, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda r: r.distance)

        match_indices = np.array([[m.queryIdx, m.trainIdx] for m in matches]).astype(np.int32)

        if match_indices.size == 0:
            return np.array([])

        # remap them back
        match_indices[:, 0] = valid_idx_i1[match_indices[:, 0]]
        match_indices[:, 1] = valid_idx_i2[match_indices[:, 1]]

        return match_indices
