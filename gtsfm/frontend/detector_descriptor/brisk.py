"""BRISK Detector-Descriptor implementation.

The detector was proposed in Leutenegger11iccv and is implemented by wrapping
over OpenCV's API.

References:
Stefan Leutenegger, Margarita Chli, and Roland Yves Siegwart.
Brisk: Binary robust invariant scalable keypoints. 
ICCV 2011. https://margaritachli.com/papers/ICCV2011paper.pdf.

Authors: John Lambert
"""
from typing import Tuple

import cv2 as cv
import numpy as np

import gtsfm.utils.features as feature_utils
import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase


class BRISKDetectorDescriptor(DetectorDescriptorBase):
    """BRISK detector-descriptor using OpenCV's implementation."""

    def apply(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Perform feature detection as well as their description.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for details about the output format.

        Args:
            image: the input image.

        Returns:
            Detected keypoints, with length N <= max_keypoints.
            Corr. descriptors, of shape (N, D) where D is the dimension of each descriptor.
        """

        # Convert to grayscale.
        gray_image = image_utils.rgb_to_gray_cv(image)

        # Create OpenCV object.
        opencv_obj = cv.BRISK_create()

        # Run the OpenCV code.
        cv_keypoints, descriptors = opencv_obj.detectAndCompute(gray_image.value_array, image.mask)

        # Convert to GTSFM's keypoints.
        keypoints = feature_utils.cast_to_gtsfm_keypoints(cv_keypoints)

        # Filter features.
        keypoints, selection_idxs = keypoints.get_top_k(self.max_keypoints)
        descriptors = descriptors[selection_idxs]

        return keypoints, descriptors
