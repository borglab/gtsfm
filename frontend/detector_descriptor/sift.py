"""SIFT Detector-Descriptor implementation.

The detector was proposed in 'Distinctive Image Features from Scale-Invariant Keypoints' and is implemented by wrapping over OpenCV's API

References:
- https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Ayush Baid
"""
from typing import Tuple

import cv2 as cv
import numpy as np

import frontend.utils.feature_utils as feature_utils
import utils.images as image_utils
from common.image import Image
from frontend.detector_descriptor.detector_descriptor_base import \
    DetectorDescriptorBase


class SIFTDetectorDescriptor(DetectorDescriptorBase):
    """SIFT detector-descriptor using OpenCV's implementation."""

    def detect_and_describe(self,
                            image: Image) -> Tuple[np.ndarray, np.ndarray]:
        """Perform feature detection as well as their description.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for
        details about the output format.

        Args:
            image: the input image.

        Returns:
            detected features as a numpy array of shape (N, 2+).
            corr. descriptors for the features, as (N, 128) sized matrix.
        """

        # conert to grayscale
        gray_image = image_utils.rgb_to_gray_cv(image)

        # Creating OpenCV object
        opencv_obj = cv.xfeatures2d.SIFT_create()

        # Run the opencv code
        cv_keypoints, descriptors = opencv_obj.detectAndCompute(
            gray_image.value_array, None)

        # convert keypoints to features
        features = feature_utils.array_of_keypoints(cv_keypoints)

        # sort the features and descriptors by the score
        sort_idx = np.argsort(-features[:, 3])[:self.max_features]
        features = features[sort_idx]
        descriptors = descriptors[sort_idx]

        return features, descriptors
