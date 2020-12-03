"""Joint detector and descriptor for the front end.

Authors: Ayush Baid
"""

import abc
from typing import List, Tuple

import dask
import numpy as np
from dask.delayed import Delayed

from common.image import Image
from common.keypoints import Keypoints


class DetectorDescriptorBase(metaclass=abc.ABCMeta):
    """Base class for all methods which provide a joint detector-descriptor to
    work on an image.

    This class serves as a combination of individual detector and descriptor.
    """

    def __init__(self, max_keypoints: int = 5000):
        """Initialize the detector-descriptor.

        Args:
            max_keypoints: Maximum number of keypoints to detect. Defaults to
                           5000.
        """
        self.max_keypoints = max_keypoints

    @abc.abstractmethod
    def detect_and_describe(self,
                            image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Perform feature detection as well as their description.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for
        details about the output format.

        Args:
            image: the input image.

        Returns:
            detected keypoints, with length N <= max_keypoints.
            corr. descriptors, of shape (N, D) where D is the dimension of each
            descriptor.
        """

    def create_computation_graph(self,
                                 image_graph: List[Delayed]
                                 ) -> Tuple[List[Delayed], List[Delayed]]:
        """
        Generates the computation graph for detections and their descriptions.

        Args:
            image_graph: computation graph for images (from a loader).

        Returns:
            List of delayed tasks for detections.
            List of delayed task for corr. descriptions.

        """
        joint_graph = [
            dask.delayed(self.detect_and_describe)(x) for x in image_graph]

        detection_graph = [x[0] for x in joint_graph]
        description_graph = [x[1] for x in joint_graph]

        return detection_graph, description_graph
