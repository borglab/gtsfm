"""Difference-of-Gaussian detector implementation.

The detector was proposed in 'Distinctive Image Features from Scale-Invariant
Keypoints' and is implemented by wrapping over OpenCV's API.

References:
- https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Ayush Baid
"""
import cv2 as cv

import gtsfm.utils.features as feature_utils
import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector.detector_base import DetectorBase


class DoG(DetectorBase):
    """DoG detector using OpenCV's implementation."""

    def detect(self, image: Image) -> Keypoints:
        """Detect the features in an image.

        Args:
            image: input image.

        Returns:
            detected keypoints, with maximum length of max_keypoints.
        """
        # init the opencv object
        opencv_obj = cv.SIFT_create()

        gray_image = image_utils.rgb_to_gray_cv(image)
        cv_keypoints = opencv_obj.detect(gray_image.value_array, None)
        keypoints = feature_utils.cast_to_gtsfm_keypoints(cv_keypoints)

        # limit number of keypoints
        keypoints, _ = keypoints.get_top_k_by_scale(self.max_keypoints)

        return keypoints
