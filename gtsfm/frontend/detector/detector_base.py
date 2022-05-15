"""Base class for the Detection stage of the frontend.

Authors: Ayush Baid
"""
import abc

import dask
from dask.delayed import Delayed

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
    def detect(self, image: Image) -> Keypoints:
        """Detect the features in an image.

        Args:
            image: input image.

        Returns:
            detected keypoints, with maximum length of max_keypoints.
        """

    def create_computation_graph(self, image_graph: Delayed) -> Delayed:
        """Generates the computation graph for performing detection.

        Args:
            image_graph: computation graph for an image.

        Returns:
            Delayed task for detection on the input image.
        """
        return dask.delayed(self.detect)(image_graph)
