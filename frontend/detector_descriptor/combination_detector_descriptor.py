""" 
Creates a combination from the separate detector and descriptor.

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
    """
    A wrapper for individual detection and description.
    """

    def __init__(self, detector: DetectorBase, descriptor: DescriptorBase):
        """
        Initialize from individual detector and descriptor.

        Args:
            detector (DetectorBase): the detector to combine
            descriptor (DescriptorBase): the descriptor to combine
        """
        self.detector = detector
        self.descriptor = descriptor

    def detect_and_describe(self, image: Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform feature detection as well as their description in a single step.

        Refer to detect() in BaseDetector and describe() in BaseDescriptor 
        for details about the output format.

        Args:
            image (Image): the input image

        Returns:
            Tuple[np.ndarray, np.ndarray]: detected features and their 
                                           descriptions as two numpy arrays
        """

        features = self.detector.detect(image)
        descriptors = self.descriptor.describe(image, features)

        return features, descriptors
