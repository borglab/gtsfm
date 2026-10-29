"""RootSIFT normalization of OpenCV's jointly detected and computed SIFT features.

Reference: Arandjelovic and Zisserman, "Three things everyone should know to improve
object retrieval", CVPR 2012. https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/
"""

from typing import Tuple

import numpy as np

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor


class RootSIFTDetectorDescriptor(SIFTDetectorDescriptor):
    """Apply RootSIFT to descriptors computed with OpenCV's original keypoint frames.

    Computing descriptors jointly with detection preserves orientation and octave
    information that would be lost in a round trip through GTSFM's Keypoints.
    """

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Detect SIFT features and L1-normalize and square-root their descriptors.

        Args:
            image: Input image, with an optional detection mask.

        Returns:
            Detected keypoints, with length N <= max_keypoints.
            Float32 descriptors of shape (N, 128), with unit L2 norm for nonzero rows.
        """
        keypoints, descriptors = super().detect_and_describe(image)
        # Use the profiled normalization formula, then ensure float32 for OpenCV matchers.
        l1_norms = np.sum(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / (l1_norms + np.finfo(float).eps)
        return keypoints, np.sqrt(descriptors).astype(np.float32)
