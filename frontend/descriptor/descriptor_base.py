"""Base class for the description stage of the frontend.

Authors: Ayush Baid
"""

import abc
from typing import List

import dask
import numpy as np
from dask.delayed import Delayed

from common.image import Image
from common.keypoints import Keypoints


class DescriptorBase(metaclass=abc.ABCMeta):
    """Base class for all the feature descriptors.

    Feature descriptors assign a vector for each input point.
    """

    @abc.abstractmethod
    def describe(self, image: Image, keypoints: Keypoints) -> np.ndarray:
        """Assign descriptors to detected features in an image.

        Arguments:
            image: the input image.
            keypoints: the keypoints to describe, of length N.

        Returns:
            the descriptors for the input features, of shape (N, D) where D is
                the dimension of each descriptor.
        """

    def create_computation_graph(self,
                                 image_graph: List[Delayed],
                                 detection_graph: List[Delayed]
                                 ) -> List[Delayed]:
        """Generates the computation graph to perform description.

        Note: two input graphs have to be of the same length.

        Args:
            image_graph: computation graph for images (from a loader).
            detection_graph: computation graph from detector, for the images in
                             image_graph.

        Returns:
            List of delayed tasks for descriptions.
        """
        assert len(image_graph) == len(detection_graph)

        return [dask.delayed(self.describe)(im, keypoints)
                for im, keypoints in zip(image_graph, detection_graph)]
