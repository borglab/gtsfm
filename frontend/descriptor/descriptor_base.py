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
        as a row vector.

        Arguments:
            image: the input image.
            features: features to describe, as a numpy array of shape (N, 2+).

        Returns:
            the descriptors for the input features, as (N, x) sized matrix.
        """

    def create_computation_graph(self,
                                 loader_graph: List[Delayed],
                                 detection_graph: List[Delayed]
                                 ) -> List[Delayed]:
        """Generates the computation graph to perform description.

        Note: two two input lists have to be of the same size.

        Args:
            loader_graph: computation graph from loader, which provides images.
            detection_graph: computation graph from detector, which provides
                             features.

        Returns:
            List[Delayed]: delayed dask elements.
        """
        assert len(loader_graph) == len(detection_graph)

        return [dask.delayed(self.describe)(im, feat)
                for im, feat in zip(loader_graph, detection_graph)]
