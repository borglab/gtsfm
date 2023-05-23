"""Base class for the Detection stage of the frontend.

Authors: Ayush Baid
"""
import abc

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints


class DetectorBase(metaclass=abc.ABCMeta):
    """Base class for all the feature detectors."""

    def __init__(self, max_keypoints: int = 5000) -> None:
        """Initialize the detector.

        Args:
            max_keypoints: Maximum number of keypoints to detect. Defaults to
                           5000.
        """
        self.max_keypoints = max_keypoints

    @abc.abstractmethod
    def apply(self, image: Image) -> Keypoints:
        """Detect the features in an image.

        Args:
            image: input image.

        Returns:
            detected keypoints, with maximum length of max_keypoints.
        """
