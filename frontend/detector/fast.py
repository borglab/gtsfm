"""FAST detector implementation.

The detector was proposed in 'Machine Learning for High-Speed Corner Detection' 
and is implemented by wrapping over OpenCV's API

References:
- https://link.springer.com/chapter/10.1007/11744023_34
- https://docs.opencv.org/3.4.2/df/d74/classcv_1_1FastFeatureDetector.html

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np

import utils.features as feature_utils
import utils.images as image_utils
from common.image import Image
from frontend.detector.detector_base import DetectorBase


class Fast(DetectorBase):
    """Fast detector using opencv's implementation."""

    def detect(self, image: Image) -> np.ndarray:
        """Detect the features in an image.

        Refer to documentation in DetectorBase for more details.

        Args:
            image: input image.

        Returns:
            detected features.
        """
        gray_image = image_utils.rgb_to_gray_cv(image.image_array)

        # init the opencv object
        opencv_obj = cv.FastFeatureDetector_create()

        cv_keypoints = opencv_obj.detect(gray_image, None)

        # sort the keypoints by score and pick top responses
        cv_keypoints = sorted(
            cv_keypoints, key=lambda x: x.response, reverse=True
        )[:self.max_features]

        # convert to numpy array
        features = feature_utils.array_of_keypoints(cv_keypoints)

        return features
