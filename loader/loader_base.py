""" Base class for Loaders.

Authors: Frank Dellaert and Ayush Baid
"""

import abc
from typing import List

import dask
import numpy as np


class LoaderBase:
    """Base class for Loaders."""

    @abc.abstractmethod
    def get_image(self, index: int) -> np.array:
        """
        Get the image at the given index

        Args:
            index (int): the index to fetch

        Returns:
            np.array: image
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        The number of images in the dataset

        Returns:
            int: the number of images
        """

    def create_computation_graph_single(self, index: int):
        """
        Creates the computation graph for a single image fetch

        Args:
            index (int): the image index

        Returns:
            [type]: [description]
        """

        return dask.delayed(self.get_image)(index)

    def create_computation_graph_all(self) -> List:
        """
        Creates the computation graph for all image fetch

        Returns:
            [type]: [description]
        """

        return [dask.delayed(self.get_image)(x) for x in range(self.__len__())]
