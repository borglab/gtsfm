"""Grid-based motion statistics (GMS) feature matching implemented as a decorator to work with other matching 
techniques.

The algorithm was proposed in "GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence" and
is implemented by calling OpenCV's API.

References:
- https://openaccess.thecvf.com/content_cvpr_2017/papers/Bian_GMS_Grid-based_Motion_CVPR_2017_paper.pdf
- https://docs.opencv.org/3.4/db/dd9/group__xfeatures2d__match.html#gaaf19e0024c555f8d8982396376150288

Authors: Ayush Baid, Travis Head
"""
from typing import List, Tuple

import cv2 as cv
import numpy as np
from cv2 import DMatch

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.matcher_base import MatcherBase

logger = logger_utils.get_logger()


class GmsMatcher(MatcherBase):
    def __init__(self, underlying_matcher_obj: MatcherBase) -> None:
        self._underlying_matcher: MatcherBase = underlying_matcher_obj
        self._with_rotation: bool = True
        self._with_scale: bool = True

    def match(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
    ) -> np.ndarray:
        matches_from_underlying_matcher = self._underlying_matcher.match(
            keypoints_i1, keypoints_i2, descriptors_i1, descriptors_i2, im_shape_i1, im_shape_i2
        )

        gms_matches = cv.xfeatures2d.matchGMS(
            size1=im_shape_i1,
            size2=im_shape_i2,
            keypoints1=keypoints_i1.cast_to_opencv_keypoints(),
            keypoints2=keypoints_i2.cast_to_opencv_keypoints(),
            matches1to2=self.__cast_to_opencv_dmatches(matches_from_underlying_matcher),
            withRotation=self._with_rotation,
            withScale=self._with_scale,
        )

        return np.array([[m.queryIdx, m.trainIdx] for m in gms_matches]).astype(np.int32)

    def __cast_to_opencv_dmatches(self, matches: np.ndarray) -> List[DMatch]:
        dmatches: List[DMatch] = []
        for idx_i1, idx_i2 in matches:
            dmatches.append(DMatch(_queryIdx=idx_i1, _trainIdx=idx_i2, _distance=0.0))

        return dmatches
