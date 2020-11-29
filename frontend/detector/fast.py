"""FAST detector implementation.

The detector was proposed in 'Machine Learning for High-Speed Corner Detection' 
and is implemented by wrapping over OpenCV's API

References:
- https://link.springer.com/chapter/10.1007/11744023_34
- https://docs.opencv.org/3.4.2/df/d74/classcv_1_1FastFeatureDetector.html

Authors: Ayush Baid
"""
import cv2 as cv

import utils.features as feature_utils
import utils.images as image_utils
from common.image import Image
from common.keypoints import Keypoints
from frontend.detector.detector_base import DetectorBase


class Fast(DetectorBase):
    """Fast detector using opencv's implementation."""

    def detect(self, image: Image) -> Keypoints:
        """Detect the features in an image.

        Coordinate system convention:
        1. The x coordinate denotes the horizontal direction (+ve direction
           towards the right).
        2. The y coordinate denotes the vertical direction (+ve direction
           downwards).
        3. Origin is at the top left corner of the image.

        Output format:
        1. If applicable, the keypoints should be sorted in decreasing order of
           score/confidence.

        Args:
            image: input image.

        Returns:
            detected keypoints, with maximum length of max_keypoints.
        """
        # init the opencv object
        opencv_obj = cv.FastFeatureDetector_create()

        gray_image = image_utils.rgb_to_gray_cv(image)
        cv_keypoints = opencv_obj.detect(gray_image.value_array, None)

        # sort the keypoints by score and pick top responses
        cv_keypoints = sorted(
            cv_keypoints, key=lambda x: x.response, reverse=True
        )[:self.max_keypoints]

        keypoints = feature_utils.cast_to_gtsfm_keypoints(cv_keypoints)

        return keypoints
