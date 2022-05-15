"""Base class for the description stage of the frontend.

Authors: Ayush Baid
"""

import abc

import dask
import numpy as np
from dask.delayed import Delayed

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints


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
            Descriptors for the input features, of shape (N, D) where D is the dimension of each descriptor.
        """

    def create_computation_graph(self, image_graph: Delayed, keypoints_graph: Delayed) -> Delayed:
        """Generates the computation graph to perform description.

        Args:
            image_graph: computation graph for an image.
            keypoints_graph: computation graph for keypoints for the image.

        Returns:
            Delayed tasks for performing description.
        """

        return dask.delayed(self.describe)(image_graph, keypoints_graph)
