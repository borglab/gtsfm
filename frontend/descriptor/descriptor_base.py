""" 
Base class for the description stage of the frontend.

Authors: Ayush Baid
"""

import abc
from typing import List

import dask
import numpy as np

from common.image import Image


class DescriptorBase(metaclass=abc.ABCMeta):
    """
    Base class for all the feature descriptors.
    Feature descriptors assign a vector for each input point.
    """

    @abc.abstractmethod
    def describe(self, image: Image, features: np.ndarray) -> np.ndarray:
        """
        Assign descriptors to detected features in an image

        Output format:
        1. Each input feature point is assigned a descriptor, which is stored as a row vector

        Arguments:
            image (Image): the input image
            features (np.ndarray): the features to describe

        Returns:
            np.ndarray: the descriptors for the input features
        """

    def create_computation_graph(self,
                                 loader_graph: List[dask.delayed],
                                 detection_graph: List[dask.delayed]) -> List[dask.delayed]:
        """
        Generates the computation graph to perform description for all the entries in the supplied dataset

        Args:
            loader_graph (List[dask.delayed]): computation graph from loader
            detection_graph (List[dask.delayed]): computation graph from detector

        Returns:
            List[dask.delayed]: delayed dask elements
        """
        return [dask.delayed(self.describe)(im, feat) for im, feat in zip(loader_graph, detection_graph)]
