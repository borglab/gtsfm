"""SIFT Descriptor implementation.

The method was proposed in 'Distinctive Image Features from Scale-Invariant
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
from frontend.descriptor.descriptor_base import DescriptorBase


class SIFTDescriptor(DescriptorBase):
    """SIFT descriptor using OpenCV's implementation."""

    def describe(self, image: Image, features: np.ndarray) -> np.ndarray:
        """Assign descriptors to features in an image.

        Output format:
        1. Each input feature point is assigned a descriptor, which is stored
        as a row vector

        Arguments:
            image: the input image.
            features: the features to describe.

        Returns:
            np.ndarray: the descriptors for the input features.
        """

        if features.size == 0:
            return np.array([])

        gray_image = image_utils.rgb_to_gray_cv(image.image_array)

        opencv_obj = cv.xfeatures2d.SIFT_create()

        # TODO(ayush): what to do about new set of keypoints
        _, descriptors = opencv_obj.compute(
            gray_image, feature_utils.keypoints_of_array(features)
        )

        return descriptors
