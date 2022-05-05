""" Base class for Loaders.

Authors: Frank Dellaert and Ayush Baid
"""

import abc
from typing import List, Optional, Tuple

import dask
from dask.delayed import Delayed
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3

import gtsfm.utils.images as img_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image


logger = logger_utils.get_logger()


class LoaderBase(metaclass=abc.ABCMeta):
    """Base class for Loaders.

    The loader provides APIs to get an image, either directly or as a dask delayed task
    """

    def __init__(self, max_resolution: int) -> None:
        """
        Args:
            max_resolution: integer representing maximum length of image's short side
               e.g. for 1080p (1920 x 1080), max_resolution would be 1080
        """
        if not isinstance(max_resolution, int):
            raise ValueError("Maximum image resolution must be an integer argument.")
        self._max_resolution = max_resolution

    def _build_image_shapes_cache(self) -> None:
        """Build a cache of the dimensions of each image in the dataset.

        We store the full-resolution dimensions, and the shape after resizing to a maximum allowed resolution.
        """
        self._image_shape_dict = {}
        self._image_shape_full_res_dict = {}
        for index in range(len(self)):
            img_full_res = self.get_image_full_res(index)
            self._image_shape_full_res_dict[index] = (img_full_res.height, img_full_res.width)
            (_, _, target_h, target_w,) = img_utils.get_downsampling_factor_per_axis(
                img_full_res.height, img_full_res.width, self._max_resolution
            )
            self._image_shape_dict[index] = (target_h, target_w)

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
    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            Image: the image at the query index.
        """

    # ignored-abstractmethod
    @abc.abstractmethod
    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3Bundler]:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

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

    def get_camera(self, index: int) -> Optional[PinholeCameraCal3Bundler]:
        """Gets the camera at the given index.

        Args:
            index: the index to fetch.

        Returns:
            Camera object with intrinsics and extrinsics, if they exist.
        """
        pose = self.get_camera_pose(index)
        intrinsics = self.get_camera_intrinsics(index)

        if pose is None or intrinsics is None:
            return None

        return PinholeCameraCal3Bundler(pose, intrinsics)

    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair. idx1 < idx2 is required.

        Note: All inherited classes should call this super method to enforce this check.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """
        return idx1 < idx2

    def get_image(self, index: int) -> Image:
        """Get the image at the given index, satisfying a maximum image resolution constraint.

        Determine how the camera intrinsics and images should be jointly rescaled based on desired img. resolution.
        Each loader implementation should set a `_max_resolution` attribute.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            Image: the image at the query index. It will be resized to satisfy the maximum
                allowed loader image resolution if the full-resolution images for a dataset
                are too large.
        """
        # No downsampling may be required, in which case target_h and target_w will be identical
        # to the full res height & width.
        img_full_res = self.get_image_full_res(index)
        if min(img_full_res.height, img_full_res.width) <= self._max_resolution:
            return img_full_res

        # Resize image.
        target_h, target_w = self._image_shape_dict[index]

        logger.info(
            "Image %d resized from (H,W)=(%d,%d) -> (%d,%d)",
            index,
            img_full_res.height,
            img_full_res.width,
            target_h,
            target_w,
        )
        resized_img = img_utils.resize_image(img_full_res, new_height=target_h, new_width=target_w)
        return resized_img

    def get_camera_intrinsics(self, index: int) -> Cal3Bundler:
        """Get the camera intrinsics at the given index, for a possibly resized image.

        Determine how the camera intrinsics and images should be jointly rescaled based on desired img. resolution.
        Each loader implementation should set a `_max_resolution` attribute.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        intrinsics_full_res = self.get_camera_intrinsics_full_res(index)
        if intrinsics_full_res is None:
            raise ValueError(f"No intrinsics found for index {index}.")

        if intrinsics_full_res.fx() <= 0:
            raise RuntimeError("Focal length must be positive.")

        if intrinsics_full_res.px() <= 0 or intrinsics_full_res.py() <= 0:
            raise RuntimeError("Principal point must have positive coordinates.")

        img_full_res_height, img_full_res_width = self.get_image_shape_full_res(index)
        # no downsampling may be required, in which case scale_u and scale_v will be 1.0
        scale_u, scale_v, _, _ = img_utils.get_downsampling_factor_per_axis(
            img_full_res_height, img_full_res_width, self._max_resolution
        )
        rescaled_intrinsics = Cal3Bundler(
            fx=intrinsics_full_res.fx() * scale_u,
            k1=0.0,
            k2=0.0,
            u0=intrinsics_full_res.px() * scale_u,
            v0=intrinsics_full_res.py() * scale_v,
        )
        return rescaled_intrinsics

    def get_image_shape_full_res(self, index: int) -> Tuple[int, int]:
        """Return the (H,W) tuple representing height, width of full-resolution image."""
        return self._image_shape_full_res_dict[index]

    def get_image_shape(self, idx: int) -> Tuple[int, int]:
        """Return a (H,W) tuple for each image"""
        image_height, image_width = self._image_shape_dict[idx]
        return (image_height, image_width)

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

    def create_computation_graph_for_cameras(self) -> Optional[List[Delayed]]:
        """Creates the computation graph for cameras.

        Returns:
            OList of delayed tasks for cameras.
        """
        N = len(self)

        if self.get_camera(0) is None:
            return None

        return [dask.delayed(self.get_camera)(i) for i in range(N)]

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
