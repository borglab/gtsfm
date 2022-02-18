"""Two way (mutual nearest neighbor) matcher.

Ref: https://github.com/colmap/colmap/blob/2b7230679957e4dccd590ab467931d6cfffb9ede/src/feature/sift.cc

Authors: Ayush Baid
"""
from typing import Dict

import cv2 as cv
import numpy as np

from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher, MatchingDistanceType


class TwoWayWithRatioTestMatcher(TwoWayMatcher):
    """Two way (mutual nearest neighbor) matcher using OpenCV."""

    def __init__(
        self,
        distance_type: MatchingDistanceType = MatchingDistanceType.EUCLIDEAN,
        ratio_test_threshold: float = 0.8,
    ):
        """Initialize the matcher.

        Args:
            distance_type: distance type for matching.
            ratio_test_threshold: ratio test threshold. Defaults to 0.8.
        """
        super().__init__()
        self._distance_type = distance_type
        self._ratio_test_threshold: float = ratio_test_threshold

    def _init_opencv_matcher(self) -> cv.DescriptorMatcher:
        """Initialize the OpenCV matcher.

        Returns:
            Matcher object.
        """
        if self._distance_type is MatchingDistanceType.EUCLIDEAN:
            distance_metric = cv.NORM_L2
        elif self._distance_type is MatchingDistanceType.HAMMING:
            distance_metric = cv.NORM_HAMMING
        else:
            raise NotImplementedError("The distance type is not in MatchingDistanceType")

        return cv.BFMatcher(normType=distance_metric, crossCheck=False)

    def _perform_matching(self, descriptors_1: np.ndarray, descriptors_2: np.ndarray) -> np.ndarray:
        """Run the core logic for matching.

        Args:
            descriptors_1: descriptors for the 1st image.
            descriptors_2: descriptors for the 2nd image.

        Returns:
            indices of the match between two images.
        """
        match_indices_1to2: Dict[int, int] = self._perform_oneway_matching(descriptors_1, descriptors_2)
        match_indices_2to1: Dict[int, int] = self._perform_oneway_matching(descriptors_2, descriptors_1)

        match_indices_1to2to1 = {
            idx1: match_indices_2to1[idx2] for idx1, idx2 in match_indices_1to2.items() if idx2 in match_indices_2to1
        }

        twoway_match_indices = np.array(
            [[idx1, idx2] for idx1, idx2 in match_indices_1to2.items() if match_indices_1to2to1.get(idx1) == idx1],
            dtype=np.uint32,
        )
        return twoway_match_indices

    def _perform_oneway_matching(self, descriptors_1: np.ndarray, descriptors_2: np.ndarray) -> Dict[int, int]:
        opencv_matcher = self._init_opencv_matcher()

        all_matches = opencv_matcher.knnMatch(descriptors_1, descriptors_2, k=2)
        matches = [m1 for m1, m2 in all_matches if m1.distance <= self._ratio_test_threshold * m2.distance]
        matches = sorted(matches, key=lambda r: r.distance)

        match_indices = {m.queryIdx: m.trainIdx for m in matches}

        return match_indices
