"""Two way (mutual nearest neighbor) matcher.

Authors: Ayush Baid
"""
from typing import Tuple, Optional

import cv2 as cv
import numpy as np
from enum import Enum
from cv2.xfeatures2d import matchGMS

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
        im_shape_i1: Tuple[int, int],  # pylint: disable=unused-argument
        im_shape_i2: Tuple[int, int],  # pylint: disable=unused-argument
        ratio: Optional[float] = 0.8,
        cross_check: bool = True,
        use_gms: bool = True,
        with_scale: bool = True,
        with_rotation: bool = True,
        threshold_factor: float = 6,
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
            im_shape_i1: shape of image #i1, as height, width.
            im_shape_i2: shape of image #i2, as height, width.
            ratio: value for Lowe's ratio test.
            cross_check: will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the
                matcher's collection is the nearest and vice versa, i.e., will only return consistent pairs.
            with_scale: take scale transformation into account in GMS.
            with_roation: take rotation transformation into account in GMS.
            threshold_factor: threshold factor for rejecting grid cells (see section 3.1 of [1]). The higher, the less
                matches.

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

        # Remove NaNs.
        valid_idx_i1 = np.nonzero(~(np.isnan(descriptors_i1).any(axis=1)))[0]
        valid_idx_i2 = np.nonzero(~(np.isnan(descriptors_i2).any(axis=1)))[0]

        descriptors_1 = descriptors_i1[valid_idx_i1]
        descriptors_2 = descriptors_i2[valid_idx_i2]

        # Run KNN matcher both ways.
        bf_matcher = cv.BFMatcher(normType=distance_metric)
        init_matches1 = bf_matcher.knnMatch(descriptors_1, descriptors_2, k=2)
        init_matches2 = bf_matcher.knnMatch(descriptors_2, descriptors_1, k=2)

        # Filter putative matches using cross check and Lowe's ratio test.
        good_matches = []
        for i in range(len(init_matches1)):
            cond = True
            if cross_check:
                cond1 = cross_check and init_matches2[init_matches1[i][0].trainIdx][0].trainIdx == i
                cond = cond and cond1
            if ratio is not None and ratio < 1:
                cond2 = init_matches1[i][0].distance <= ratio * init_matches1[i][1].distance
                cond = cond and cond2
            if cond:
                good_matches.append(init_matches1[i][0])
        matches = sorted(good_matches, key=lambda r: r.distance)

        # Verify putative matches with GMS.
        if use_gms:
            matches = matchGMS(
                im_shape_i1,
                im_shape_i2,
                keypoints_i1.cast_to_opencv_keypoints(),
                keypoints_i2.cast_to_opencv_keypoints(),
                matches,
                withScale=with_scale,
                withRotation=with_rotation,
                thresholdFactor=threshold_factor,
            )

        # Compile match indices.
        match_indices = np.array([[m.queryIdx, m.trainIdx] for m in matches]).astype(np.int32)
        if match_indices.size == 0:
            return np.array([])

        # Remap them back
        match_indices[:, 0] = valid_idx_i1[match_indices[:, 0]]
        match_indices[:, 1] = valid_idx_i2[match_indices[:, 1]]

        return match_indices
