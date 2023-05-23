"""FAST detector implementation.

The detector was proposed in 'Machine Learning for High-Speed Corner Detection' 
and is implemented by wrapping over OpenCV's API.

References:
- https://link.springer.com/chapter/10.1007/11744023_34
- https://docs.opencv.org/3.4.2/df/d74/classcv_1_1FastFeatureDetector.html

Authors: Ayush Baid
"""
import cv2 as cv

import gtsfm.utils.features as feature_utils
import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector.detector_base import DetectorBase


class Fast(DetectorBase):
    """Fast detector using OpenCV's implementation."""

    def apply(self, image: Image) -> Keypoints:
        """Detect the features in an image.

        Args:
            image: input image.

        Returns:
            detected keypoints, with maximum length of max_keypoints.
        """
        # init the opencv object
        opencv_obj = cv.FastFeatureDetector_create()

        gray_image = image_utils.rgb_to_gray_cv(image)
        cv_keypoints = opencv_obj.detect(gray_image.value_array, None)
        keypoints = feature_utils.cast_to_gtsfm_keypoints(cv_keypoints)

        # limit number of keypoints
        keypoints, _ = keypoints.get_top_k(self.max_keypoints)

        return keypoints
