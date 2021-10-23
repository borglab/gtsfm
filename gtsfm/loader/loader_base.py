""" Base class for Loaders.

Authors: Frank Dellaert and Ayush Baid
"""

import abc
from typing import Dict, List, Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3

import gtsfm.utils.images as img_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints


logger = logger_utils.get_logger()

# TODO(Ayush): figure out how to use value from the config here
SAMPSON_ERROR_EPIPOLAR_DISTANCE_THRESHOLD = 4


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
        # no downsampling may be required, in which case target_h and target_w will be identical
        # to the full res height & width.
        img_full_res = self.get_image_full_res(index)
        (
            _,
            _,
            target_h,
            target_w,
        ) = img_utils.get_downsampling_factor_per_axis(img_full_res.height, img_full_res.width, self._max_resolution)
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

        img_full_res = self.get_image_full_res(index)
        # no downsampling may be required, in which case scale_u and scale_v will be 1.0
        scale_u, scale_v, _, _ = img_utils.get_downsampling_factor_per_axis(
            img_full_res.height, img_full_res.width, self._max_resolution
        )
        rescaled_intrinsics = Cal3Bundler(
            fx=intrinsics_full_res.fx() * scale_u,
            k1=0.0,
            k2=0.0,
            u0=intrinsics_full_res.px() * scale_u,
            v0=intrinsics_full_res.py() * scale_v,
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

    def evaluate_correspondences(
        self, i1: int, i2: int, keypoints_i1: Keypoints, keypoints_i2: Keypoints
    ) -> Optional[np.ndarray]:
        """Evaluate the correspondences by using ground truth geometry/object model.

        Args:
            i1: index of the first image.
            i2: index of the second image.
            keypoints_i1: keypoints for the correspondences in i1, of length N.
            keypoints_i2: keypoints for the correspondences in i2, of length N.

        Returns:
            Boolean numpy array indicating the correctness of the correspondences, of shape (N, ).
        """
        wTi1 = self.get_camera_pose(i1)
        wTi2 = self.get_camera_pose(i2)
        if wTi1 is None or wTi2 is None:
            return None

        i2Ti1 = wTi2.between(wTi1)
        return metric_utils.count_correct_correspondences(
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            intrinsics_i1=self.get_camera_intrinsics(i1),
            intrinsics_i2=self.get_camera_intrinsics(i2),
            i2Ti1=i2Ti1,
            epipolar_dist_threshold=SAMPSON_ERROR_EPIPOLAR_DISTANCE_THRESHOLD,
        )

    def create_computation_graph_for_correspondence_eval_function(self) -> Dict[Tuple[int, int], Delayed]:
        """Creates the delayed `evaluate_correspondences` function nodes for each valid pair.

        Returns:
            The delayed objects as a dict, keyed by the indices of image pairs.
        """
        return {
            (idx1, idx2): dask.delayed(
                lambda keypoints_i1, keypoints_i2, i1=idx1, i2=idx2: self.evaluate_correspondences(
                    i1, i2, keypoints_i1, keypoints_i2
                )
            )
            for idx1, idx2 in self.get_valid_pairs()
        }
