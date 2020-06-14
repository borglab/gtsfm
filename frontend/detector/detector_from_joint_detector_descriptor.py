"""
A wrapper over joint detector-descriptors so that we can just use it as a descriptor.

Authors: Ayush Baid
"""

import numpy as np

from common.image import Image
from frontend.detector.detector_base import DetectorBase
from frontend.detector_descriptor.detector_descriptor_base import \
    DetectorDescriptorBase


class DetectorFromDetectorDescriptor(DetectorBase):
    """
    A wrapper class to just expose the Detector component of DetectorDescriptor.
    """

    def __init__(self, detector_descriptor: DetectorDescriptorBase):
        """
        Initialize a Detector from a joint detector descriptor.

        Args:
            detector_descriptor (DetectorDescriptorBase): the joint detector descriptor
        """
        self.detector_descriptor = detector_descriptor

    def detect(self, image: Image) -> np.ndarray:
        """
        Detect the features on the input image.

        Refer to documentation in DetectorBase for more details.

        Args:
            image (Image): input image

        Returns:
            np.ndarray: detected features
        """
        features, _ = self.detector_descriptor.detect_and_describe(image)

        return features
