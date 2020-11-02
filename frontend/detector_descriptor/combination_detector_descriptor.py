""" 
Joint detector-description combination from stand-alone detector and descriptor.

Authors: Ayush Baid
"""
from typing import Tuple

import numpy as np

from common.image import Image
from frontend.descriptor.descriptor_base import DescriptorBase
from frontend.detector.detector_base import DetectorBase
from frontend.detector_descriptor.detector_descriptor_base import \
    DetectorDescriptorBase


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

    def detect_and_describe(self,
                            image: Image) -> Tuple[np.ndarray, np.ndarray]:
        """Perform feature detection as well as their description.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for
        details about the output format.

        Args:
            image: the input image.

        Returns:
            detected features and their descriptions as two numpy arrays.
        """

        features = self.detector.detect(image)
        descriptors = self.descriptor.describe(image, features)

        return features, descriptors
