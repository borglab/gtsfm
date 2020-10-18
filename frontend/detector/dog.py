"""Difference-of-Gaussian detector implementation.

The detector was proposed in 'Distinctive Image Features from Scale-Invariant
Keypoints' and is implemented by wrapping over OpenCV's API

References:
- https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np

import utils.features as feature_utils
import utils.images as image_utils
from common.image import Image
from frontend.detector.detector_base import DetectorBase


class DoG(DetectorBase):
    """DoG detector using opencv's implementation."""

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
        opencv_obj = cv.xfeatures2d.SIFT_create()

        cv_keypoints = opencv_obj.detect(gray_image, None)

        # sort the keypoints by score and pick top responses
        cv_keypoints = sorted(
            cv_keypoints, key=lambda x: x.response, reverse=True
        )[:self.max_features]

        # convert to numpy array
        features = feature_utils.array_of_keypoints(cv_keypoints)

        return features
