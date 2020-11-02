"""Base class for the description stage of the frontend.

Authors: Ayush Baid
"""

import abc
from typing import List

import dask
import numpy as np
from dask.delayed import Delayed

from common.image import Image


class DescriptorBase(metaclass=abc.ABCMeta):
    """Base class for all the feature descriptors.

    Feature descriptors assign a vector for each input point.
    """

    @abc.abstractmethod
    def describe(self, image: Image, features: np.ndarray) -> np.ndarray:
        """Assign descriptors to detected features in an image.

        Output format:
        1. Each input feature point is assigned a descriptor, which is stored
        as a row vector

        Arguments:
            image: the input image.
            features: the features to describe.

        Returns:
            np.ndarray: the descriptors for the input features, as rows
        """

    def create_computation_graph(self,
                                 loader_graph: List[Delayed],
                                 detection_graph: List[Delayed]
                                 ) -> List[Delayed]:
        """Generates the computation graph to perform description.
        Args:
            loader_graph: computation graph from loader, which provides images.
            detection_graph: computation graph from detector, which provides
                             features.

        Returns:
            List[Delayed]: delayed dask elements
        """
        return [dask.delayed(self.describe)(im, feat)
                for im, feat in zip(loader_graph, detection_graph)]
