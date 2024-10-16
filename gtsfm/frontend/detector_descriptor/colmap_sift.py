"""SIFT Detector-Descriptor implementation.

The detector was proposed in 'Distinctive Image Features from Scale-Invariant Keypoints' and is implemented by wrapping
over OpenCV's API.

References:
- https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Ayush Baid
"""
from typing import Tuple

import numpy as np
import pycolmap

import gtsfm.utils.features as feature_utils
import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase


class ColmapSIFTDetectorDescriptor(DetectorDescriptorBase):
    """SIFT detector-descriptor using OpenCV's implementation."""

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
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

        # Create OpenCV object every time as the object is not pickle-able.
        # TODO (travisdriver): Add GPU support
        colmap_obj = pycolmap.Sift()

        # Extract features.
        features, descriptors = colmap_obj.extract(gray_image.value_array, max_num_features=self.max_keypoints)

        # Convert to GTSFM's keypoints.
        # Note: Columns of features is x-coordinate, y-coordinate, scale, and orientation, respectively.
        keypoints = Keypoints(coordinates=features[..., :2], scales=features[:, 2])

        return keypoints, descriptors
