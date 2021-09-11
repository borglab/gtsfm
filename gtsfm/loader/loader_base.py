""" Base class for Loaders.

Authors: Frank Dellaert and Ayush Baid
"""

import abc
from typing import List, Optional, Tuple

import dask
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Pose3

import gtsfm.utils.images as img_utils
import gtsfm.utils.io as io_utils
from gtsfm.common.image import Image


class LoaderBase(metaclass=abc.ABCMeta):
    """Base class for Loaders.

    The loader provides APIs to get an image, either directly or as a dask delayed task
    """

    def __init__(self) -> None:
        """
        Each loader implementation should set a `_max_resolution` attribute, that is
        used to determine how the camera intrinsics and images should be jointly rescaled.
        """
        if not hasattr(self, "_max_resolution"):
            raise RuntimeError("Each loader implementation must set the maximum image resolution for inference.")

        # read one image, to check if we need to downsample the images
        img = io_utils.load_image(self._image_paths[0])
        sample_h, sample_w = img.height, img.width
        # no downsampling may be required, in which case scale_u and scale_v will be 1.0
        (
            self._scale_u,
            self._scale_v,
            self._target_h,
            self._target_w,
        ) = img_utils.get_downsampling_factor_per_axis(sample_h, sample_w, self._max_resolution)

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
    def get_image_native_resolution(self, index: int) -> Image:
        """
        Get the image at the given index, at native resolution.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            Image: the image at the query index.
        """

    # ignored-abstractmethod
    @abc.abstractmethod
    def get_camera_intrinsics_native_resolution(self, index: int) -> Optional[Cal3Bundler]:
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

    def get_image(self, index: int) -> Image:
        """Get the image at the given index, for possibly resized image.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            Image: the image at the query index.
        """
        img_native = self.get_image_native_resolution(index)
        resized_img = img_utils.resize_image(img_native, new_height=self._target_h, new_width=self._target_w)
        return resized_img

    def get_camera_intrinsics(self, index: int) -> Cal3Bundler:
        """Get the camera intrinsics at the given index, for possibly resized image.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        intrinsics_native = self.get_camera_intrinsics_native_resolution(index)
        rescaled_intrinsics = Cal3Bundler(
            fx=intrinsics_native.fx() * self._scale_u,
            k1=0.0,
            k2=0.0,
            u0=intrinsics_native.px() * self._scale_u,
            v0=intrinsics_native.py() * self._scale_v,
        )
        return rescaled_intrinsics

    def get_image_shape(self, idx: int) -> Tuple[int, int]:
        """Return a (H,W) tuple for each image"""
        image = self.get_image(idx)
        return (image.height, image.width)

    def create_computation_graph_for_images(self) -> List[Delayed]:
        """Creates the computation graph for image fetches.

        Returns:
            list of delayed tasks for images.
        """
        N = len(self)

        return [dask.delayed(self.get_image)(x) for x in range(N)]

    def create_computation_graph_for_intrinsics(self) -> List[Delayed]:
        """Creates the computation graph for camera intrinsics.

        Returns:
            list of delayed tasks for camera intrinsics.
        """
        N = len(self)

        return [dask.delayed(self.get_camera_intrinsics)(x) for x in range(N)]

    def create_computation_graph_for_poses(self) -> Optional[List[Delayed]]:
        """Creates the computation graph for camera poses.

        Returns:
            list of delayed tasks for camera poses.
        """
        N = len(self)

        if self.get_camera_pose(0) is None:
            # if the 0^th pose is None, we assume none of the pose are available
            return None

        return [dask.delayed(self.get_camera_pose)(x) for x in range(N)]

    def create_computation_graph_for_image_shapes(self) -> List[Delayed]:
        """Creates the computation graph for image shapes.

        Returns:
            list of delayed tasks for image shapes.
        """
        N = len(self)
        return [dask.delayed(self.get_image_shape)(x) for x in range(N)]

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
