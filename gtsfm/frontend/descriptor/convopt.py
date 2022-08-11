"""ConvOpt Descriptor implementation.

The method was proposed in 'Learning Local Feature Descriptors Using Convex Optimisation' and is implemented by
wrapping over OpenCV's API.

Note: this is a standalone descriptor.

References:
https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/simonyan14learning.pdf

Authors: John Lambert
"""
import cv2 as cv2
import numpy as np

import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.descriptor.descriptor_base import DescriptorBase


class ConvOptDescriptor(DescriptorBase):
    """ConvOpt descriptor using OpenCV's implementation."""

    def describe(self, image: Image, keypoints: Keypoints) -> np.ndarray:
        """Assign descriptors to detected features in an image.

        Arguments:
            image: the input image.
            keypoints: Keypoints object representing N keypoints for which to form descriptors.

        Returns:
            Descriptors for the input features, of shape (N, D) where D is the dimension of each descriptor.
        """
        if len(keypoints) == 0:
            return np.array([])

        gray_image = image_utils.rgb_to_gray_cv(image)

        opencv_obj = cv2.xfeatures2d.VGG_create()
        _, descriptors = opencv_obj.compute(gray_image.value_array, keypoints.cast_to_opencv_keypoints())

        return descriptors
