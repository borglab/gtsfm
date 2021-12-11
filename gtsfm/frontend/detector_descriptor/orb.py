"""ORB detector-descriptor implementation.

The algorithm was proposed in 'Orb: an efficient alternative to sift or surf' and is implemented by wrapping over 
OpenCV's API.

References:
- https://ieeexplore.ieee.org/document/6126544
- https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html

Authors: Ayush Baid
"""
from typing import Tuple

import cv2 as cv
import numpy as np

import gtsfm.utils.features as feature_utils
import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase


class ORB(DetectorDescriptorBase):
    """DoG detector using OpenCV's implementation, with settings optimized for GMS."""

    def __init__(self, max_keypoints: int = 10000, fast_threshold: int = 0) -> None:
        super().__init__(max_keypoints=max_keypoints)
        self._fast_threshold = fast_threshold

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Perform feature detection as well as their description.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for details about the output format.

        Args:
            image: the input image.

        Returns:
            Detected keypoints, with length N <= max_keypoints.
            Corr. descriptors, of shape (N, D) where D is the dimension of each descriptor.
        """
        gray_image = image_utils.rgb_to_gray_cv(image)

        opencv_obj = cv.ORB_create(nfeatures=self.max_keypoints, fastThreshold=self._fast_threshold)
        cv_keypoints, descriptors = opencv_obj.detectAndCompute(gray_image.value_array, image.mask)

        keypoints = feature_utils.cast_to_gtsfm_keypoints(cv_keypoints)

        keypoints, selection_idxs = keypoints.get_top_k(self.max_keypoints)
        descriptors = descriptors[selection_idxs]

        return keypoints, descriptors
