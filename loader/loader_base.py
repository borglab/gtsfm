""" Base class for Loaders.

Authors: Frank Dellaert and Ayush Baid
"""

import abc
from typing import List, Optional

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Pose3

from common.image import Image


class LoaderBase(metaclass=abc.ABCMeta):
    """Base class for Loaders.

    The loader provides APIs to get an image, either directly or as a dask delayed task
    """

    # ignored-abstractmethod
    @abc.abstractmethod
    def __len__(self) -> int:
        """
        The number of images in the dataset.

        Returns:
            the number of images.
        """

    # ignored-abstractmethod
    @abc.abstractmethod
    def get_image(self, index: int) -> Image:
        """
        Get the image at the given index.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            Image: the image at the query index.
        """

    # ignored-abstractmethod
    @abc.abstractmethod
    def get_camera_intrinsics(self, index: int) -> Optional[Cal3Bundler]:
        """Get the camera intrinsics at the given index.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """

    @abc.abstractmethod
    def get_geometry(self, idx1: int, idx2: int) -> Optional[np.ndarray]:
        """Get the ground truth essential matrix/homography that maps
        measurement in image #idx1 to points/lines in #idx2.

        The function returns either idx2_E_idx1 or idx2_H_idx1.

        Args:
            idx1: one of image indices.
            idx2: one of image indices.

        Returns:
            essential matrix/homography matrix.
        """

    # ignored-abstractmethod
    @abc.abstractmethod
    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            the camera pose w_P_index.
        """

    @abc.abstractmethod
    def validate_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
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
