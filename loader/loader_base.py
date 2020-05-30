""" Base class for Loaders.

Authors: Frank Dellaert and Ayush Baid
"""

import abc
from typing import List

import dask

from common.image import Image


class LoaderBase(metaclass=abc.ABCMeta):
    """Base class for Loaders.

    The loader provides APIs to get an image, either directly or as a dask delayed task
    """

    # ignored-abstractmethod
    @abc.abstractmethod
    def __len__(self) -> int:
        """
        The number of images in the dataset

        Returns:
            int: the number of images
        """

    # ignored-abstractmethod
    @abc.abstractmethod
    def get_image(self, index: int) -> Image:
        """
        Get the image at the given index

        Args:
            index (int): the index to fetch

        Returns:
            Image: the image at the query index
        """

    def delayed_get_image(self, index: int) -> dask.delayed:
        """
        Wraps the get_image evaluation in a dask.delayed

        Args:
            index (int): the image index

        Returns:
            dask.delayed: the get_image function for the given index wrapped in dask.delayed
        """
        # TODO(ayush): is it the correct return type

        return dask.delayed(self.get_image)(index)

    def create_computation_graph(self) -> List:
        """
        Creates the computation graph for all image fetches

        Returns:
           List: list of dask's Delayed object
        """

        return [self.delayed_get_image(x) for x in range(self.__len__())]
