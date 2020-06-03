"""
Fast Dectector

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np

import utils.feature_utils as feature_utils
import utils.image_utils as image_utils
from common.image import Image
from frontend.detector.detector_base import DetectorBase


class Fast(DetectorBase):
    """
    Fast detector using opencv's implementation
    """

    def __init__(self):
        super().__init__()

        # init the opencv object
        self.opencv_obj = cv.FastFeatureDetector_create()

    def detect(self, image: Image):
        gray_image = image_utils.rgb_to_gray_cv(image.image_array)

        cv_keypoints = self.opencv_obj.detect(gray_image, None)

        # sort the keypoints by score
        cv_keypoints = sorted(
            cv_keypoints, key=lambda x: x.response, reverse=True)

        # convert to numpy array
        features = feature_utils.convert_from_opencv_keypoints(cv_keypoints)

        return features
