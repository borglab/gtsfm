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

        Coordinate system convention:
        1. The x coordinate denotes the horizontal direction (+ve direction
           towards the right).
        2. The y coordinate denotes the vertical direction (+ve direction
           downwards).
        3. Origin is at the top left corner of the image.

        Output format:
        1. If applicable, the keypoints should be sorted in decreasing order of
           score/confidence.

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
