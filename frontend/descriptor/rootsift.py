"""
RootSIFT descriptor implementation.

This descriptor was proposed in 'Three things everyone should know to improve object retrieval' and is build upon OpenCV's SIFT descriptor.

References: 
- https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np

import frontend.utils.feature_utils as feature_utils
import utils.image_utils as image_utils
from common.image import Image
from frontend.descriptor.descriptor_base import DescriptorBase


class RootSIFT(DescriptorBase):
    """
    RootSIFT descriptor using OpenCV's implementation
    """

    def __init__(self):
        super().__init__()

    def describe(self, image: Image, features: np.ndarray) -> np.ndarray:
        """
        Assign descriptors to detected features in an image

        Arguments:
            image (Image): the input image
            features (np.ndarray): the features to describe

        Returns:
            np.ndarray: the descriptors for the input features
        """

        # check if we have valid features to operate on
        if features.size == 0:
            return np.array([])

        gray_image = image_utils.rgb_to_gray_cv(image.image_array)

        # init the opencv object
        opencv_obj = cv.xfeatures2d.SIFT_create()

        # TODO(ayush): what to do about new set of keypoints
        _, sift_desc = opencv_obj.compute(
            gray_image, feature_utils.keypoints_of_array(features)
        )

        # Step 1: L1 normalization
        sift_desc = sift_desc / \
            (np.sum(sift_desc, axis=1, keepdims=True)+1e-8)

        # Step 2: Element wise square-root
        sift_desc = np.sqrt(sift_desc)

        # Step 3: L2 normalization
        sift_desc = sift_desc / \
            (np.linalg.norm(sift_desc, axis=1, keepdims=True)+1e-8)

        return sift_desc
