"""RootSIFT  Detector-Descriptor implementation.

The detector was proposed in 'Distinctive Image Features from Scale-Invariant Keypoints' and is implemented by wrapping
over OpenCV's API. The descriptor was proposed in 'Three things everyone should know to improve object retrieval' and is built upon OpenCV's API.

References:
- https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
- https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Ayush Baid
"""
from typing import Tuple

import numpy as np

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor


class RootSIFTDetectorDescriptor(SIFTDetectorDescriptor):
    """RootSIFT detector-descriptor using OpenCV's implementation."""

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Perform feature detection as well as their description.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for details about the output format.

        Args:
            image: the input image.

        Returns:
            Detected keypoints, with length N <= max_keypoints.
            Corr. descriptors, of shape (N, D) where D is the dimension of each descriptor.
        """

        keypoints, sift_desc = super().detect_and_describe(image)

        # Step 1: L1 normalization
        l1_norms = np.sum(sift_desc, axis=1, keepdims=True)
        rootsift_desc = sift_desc / (l1_norms + np.finfo(float).eps)

        # Step 2: Element wise square-root
        rootsift_desc = np.sqrt(rootsift_desc)

        return keypoints, rootsift_desc
