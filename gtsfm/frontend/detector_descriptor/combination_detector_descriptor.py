""" 
Joint detector-description combination from stand-alone detector and descriptor.

Authors: Ayush Baid
"""
from typing import Tuple

import numpy as np

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.descriptor.descriptor_base import DescriptorBase
from gtsfm.frontend.detector.detector_base import DetectorBase
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase


class CombinationDetectorDescriptor(DetectorDescriptorBase):
    """A wrapper to combine stand-alone detection and description."""

    def __init__(self, detector: DetectorBase, descriptor: DescriptorBase):
        """Initialize from individual detector and descriptor.

        Args:
            detector: the detector to combine.
            descriptor: the descriptor to combine.
        """
        self.detector = detector
        self.descriptor = descriptor

    def apply(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Perform feature detection as well as their description.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for
        details about the output format.

        Args:
            image: the input image.

        Returns:
            Detected keypoints, with length N <= max_keypoints.
            Corr. descriptors, of shape (N, D) where D is the dimension of each descriptor.
        """
        keypoints = self.detector.apply(image)
        descriptors = self.descriptor.apply(image, keypoints)

        return keypoints, descriptors
