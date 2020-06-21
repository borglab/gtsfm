"""
FAST detector implementation.

The detector was proposed in 'Machine Learning for High-Speed Corner Detection' and is implemented by wrapping over OpenCV's API

References:
- https://link.springer.com/chapter/10.1007/11744023_34
- https://docs.opencv.org/3.4.2/df/d74/classcv_1_1FastFeatureDetector.html

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np

import frontend.utils.feature_utils as feature_utils
import utils.image_utils as image_utils
from common.image import Image
from frontend.detector.detector_base import DetectorBase


class Fast(DetectorBase):
    """Fast detector using opencv's implementation"""

    def __init__(self):
        super().__init__()

        # init the opencv object
        self.opencv_obj = cv.FastFeatureDetector_create()

    def detect(self, image: Image) -> np.ndarray:
        """
        Detect the features in an image.

        Refer to the documentation in DetectorBase for more details.

        Arguments:
            image (Image): the input RGB image as a 3D numpy array

        Returns:
            np.ndarray: detected features as a numpy array
        """
        gray_image = image_utils.rgb_to_gray_cv(image.image_array)

        cv_keypoints = self.opencv_obj.detect(gray_image, None)

        # sort the keypoints by score
        cv_keypoints = sorted(
            cv_keypoints, key=lambda x: x.response, reverse=True)

        # convert to numpy array
        features = feature_utils.array_of_keypoints(cv_keypoints)

        return features
