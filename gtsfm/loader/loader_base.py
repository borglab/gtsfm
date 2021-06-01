""" Base class for Loaders.

Authors: Frank Dellaert and Ayush Baid
"""

import abc
from typing import List, Optional, Tuple

import dask
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Pose3

from gtsfm.common.image import Image


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
    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """

    def create_computation_graph_for_images(self) -> List[Delayed]:
        """Creates the computation graph for image fetches.

        Returns:
            list of delayed tasks for images.
        """
        N = self.__len__()

        return [dask.delayed(self.get_image)(x) for x in range(N)]

    def create_computation_graph_for_intrinsics(self) -> List[Delayed]:
        """Creates the computation graph for camera intrinsics.

        Returns:
            list of delayed tasks for camera intrinsics.
        """
        N = self.__len__()

        return [dask.delayed(self.get_camera_intrinsics)(x) for x in range(N)]

    def create_computation_graph_for_poses(self) -> Optional[List[Delayed]]:
        """Creates the computation graph for camera poses.

        Returns:
            list of delayed tasks for camera poses.
        """
        N = self.__len__()

        if self.get_camera_pose(0) is None:
            # if the 0^th pose is None, we assume none of the pose are available
            return None

        return [dask.delayed(self.get_camera_pose)(x) for x in range(N)]

    def get_valid_pairs(self) -> List[Tuple[int, int]]:
        """Get the valid pairs of images for this loader.

        Returns:
            list of valid index pairs.
        """
        indices = []

        for idx1 in range(self.__len__()):
            for idx2 in range(self.__len__()):
                if self.is_valid_pair(idx1, idx2):
                    indices.append((idx1, idx2))

        return indices
