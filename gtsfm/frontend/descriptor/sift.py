"""SIFT Descriptor implementation.

The method was proposed in 'Distinctive Image Features from Scale-Invarian Keypoints' and is implemented by wrapping
over OpenCV's API.

Note: this is a standalone descriptor.

References:
- https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np

import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.descriptor.descriptor_base import DescriptorBase


class SIFTDescriptor(DescriptorBase):
    """SIFT descriptor using OpenCV's implementation."""

    def describe(self, image: Image, keypoints: Keypoints) -> np.ndarray:
        """Assign descriptors to detected features in an image.

        Arguments:
            image: the input image.
            keypoints: the keypoints to describe, of length N.

        Returns:
            Descriptors for the input features, of shape (N, D) where D is the dimension of each descriptor.
        """
        if len(keypoints) == 0:
            return np.array([])

        gray_image = image_utils.rgb_to_gray_cv(image)

        opencv_obj = cv.SIFT_create()

        # TODO(ayush): what to do about new set of keypoints
        _, descriptors = opencv_obj.compute(gray_image.value_array, keypoints.cast_to_opencv_keypoints())

        return descriptors
