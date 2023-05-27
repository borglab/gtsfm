"""RootSIFT descriptor implementation.

This descriptor was proposed in 'Three things everyone should know to improve object retrieval' and is build upon
OpenCV's API.

Note: this is a standalone descriptor.

References:
- https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Ayush Baid
"""
import numpy as np

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.descriptor.sift import SIFTDescriptor


class RootSIFTDescriptor(SIFTDescriptor):
    """RootSIFT descriptor using OpenCV's implementation."""

    def apply(self, image: Image, keypoints: Keypoints) -> np.ndarray:
        """Assign descriptors to detected features in an image.

        Note: Each descriptor will have unit L2-norm, as L1-normalization followed by a square root already gives unit
        norm under the L2 norm.

        Arguments:
            image: the input image.
            keypoints: the keypoints to describe, of length N.

        Returns:
            Descriptors for the input features, of shape (N, D) where D is the dimension of each descriptor.
        """
        if len(keypoints) == 0:
            return np.array([])

        sift_desc = super().apply(image, keypoints)

        # Step 1: L1 normalization
        l1_norms = np.sum(sift_desc, axis=1, keepdims=True)
        sift_desc = sift_desc / (l1_norms + np.finfo(float).eps)

        # Step 2: Element wise square-root
        sift_desc = np.sqrt(sift_desc)

        return sift_desc
