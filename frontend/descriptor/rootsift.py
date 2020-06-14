"""
RootSIFT Dectector

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

        # init the opencv object
        self.opencv_obj = cv.xfeatures2d.SIFT_create()

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

        # TODO(ayush): what to do about new set of keypoints
        _, sift_desc = self.opencv_obj.compute(
            gray_image, feature_utils.convert_to_opencv_keypoints(features)
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
