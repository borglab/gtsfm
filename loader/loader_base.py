""" Base class for Loaders.

Authors: Frank Dellaert and Ayush Baid
"""

import abc
from typing import List, Optional

import dask
import numpy as np
from dask.delayed import Delayed

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

    # ignored-abstractmethod
    @abc.abstractmethod
    def get_camera_intrinsics(self, index: int) -> np.ndarray:
        """Get the camera intrinsics at the given index.

        Args:
            index (int): the index to fetch

        Returns:
            np.ndarray: the 3x3 intrinsics matrix of the camera
        """

    @abc.abstractmethod
    def get_geometry(self, idx1: int, idx2: int) -> Optional[np.ndarray]:
        """Get the ground truth fundamental matrix/homography that maps
        measurement from image #idx2 to points/lines in idx1.

        The function returns either idx1_F_idx2 or idx1_H_idx2.

        Args:
            idx1: one of image indices.
            idx2: one of image indices.

        Returns:
            fundamental matrix/homography matrix
        """

    @abc.abstractmethod
    def get_camera_extrinsics(self, index: int) -> Optional[np.ndarray]:
        """Get the camera extrinsics (pose) at the given index.

        The extrinsics format is [wRc, wTc]

        Args:
            index: the index to fetch.

        Returns:
            the 3x4 extrinsics matrix of the camera.
        """

    def delayed_get_image(self, index: int) -> Delayed:
        """
        Wraps the get_image evaluation in a dask.delayed

        Args:
            index (int): the image index

        Returns:
            Delayed: the get_image function for the given index wrapped in dask.delayed
        """
        return dask.delayed(self.get_image)(index)

    def create_computation_graph(self) -> List[Delayed]:
        """
        Creates the computation graph for all image fetches

        Returns:
           List[Delayed]: list of delayed image loading
        """

        return [self.delayed_get_image(x) for x in range(self.__len__())]
