"""Base class for the Detection stage of the frontend.

Authors: Ayush Baid
"""
import abc
from typing import List

import dask
from dask.delayed import Delayed

from common.image import Image
from common.keypoints import Keypoints


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

    def create_computation_graph(self,
                                 image_graph: List[Delayed]
                                 ) -> List[Delayed]:
        """Generates the computation graph for performing detections.
        Args:
            image_graph: computation graph for images (from a loader).

        Returns:
            List of delayed tasks for detection.
        """
        return [dask.delayed(self.detect)(x) for x in image_graph]
