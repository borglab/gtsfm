"""A wrapper over joint detector-descriptor to convert it to a detector.

Authors: Ayush Baid
"""
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector.detector_base import DetectorBase
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase


class DetectorFromDetectorDescriptor(DetectorBase):
    """A wrapper class to expose the Detector component of a DetectorDescriptor.

    Performs the joint detection and description but returns only the keypoints.
    """

    def __init__(self, detector_descriptor: DetectorDescriptorBase):
        """Initialize a detector from a joint detector descriptor.

        Args:
            detector_descriptor: joint detector descriptor.
        """
        super().__init__()

        self.detector_descriptor = detector_descriptor

    def detect(self, image: Image) -> Keypoints:
        """Detect the features in an image.

        Args:
            image: input image.

        Returns:
            detected keypoints, with maximum length of max_keypoints.
        """
        features, _ = self.detector_descriptor.apply(image)

        return features
