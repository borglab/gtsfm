"""RootSIFT descriptor implementation.

This descriptor was proposed in 'Three things everyone should know to improve
object retrieval' and is build upon OpenCV's SIFT descriptor.

Note: this is a descriptor

References: 
- https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Ayush Baid
"""
import numpy as np

from common.image import Image
from frontend.descriptor.sift import SIFTDescriptor


class RootSIFTDescriptor(SIFTDescriptor):
    """RootSIFT descriptor using OpenCV's implementation."""

    def describe(self, image: Image, features: np.ndarray) -> np.ndarray:
        """Assign descriptors to features in an image.

        Output format:
        1. Each input feature point is assigned a descriptor, which is stored
        as a row vector.

        Arguments:
            image: the input image.
            features: features to describe, as a numpy array of shape (N, 2+).

        Returns:
            the descriptors for the input features, as (N, 128) sized matrix.
        """
        if features.size == 0:
            return np.array([])
          
        sift_desc = super().describe(image, features)

        # Step 1: L1 normalization
        sift_desc = sift_desc / \
            (np.sum(sift_desc, axis=1, keepdims=True)+1e-8)

        # Step 2: Element wise square-root
        sift_desc = np.sqrt(sift_desc)

        # Step 3: L2 normalization
        sift_desc = sift_desc / \
            (np.linalg.norm(sift_desc, axis=1, keepdims=True)+1e-8)

        return sift_desc
