""" Base class for Loaders.

Authors: Frank Dellaert and Ayush Baid
"""

import abc
import logging
from typing import Dict, List, Optional, Tuple

import dask
from dask.delayed import Delayed
from dask.distributed import Client, Future
from gtsam import Cal3Bundler, Cal3_S2, Cal3DS2, Pose3
from trimesh import Trimesh

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.images as img_utils
import gtsfm.utils.io as io_utils
from gtsfm.common.image import Image
from gtsfm.common.pose_prior import PosePrior
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata

logger = logging.getLogger(__name__)


class LoaderBase(GTSFMProcess):
    """Base class for Loaders.

    The loader provides APIs to get an image, either directly or as a dask delayed task
    """

    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        # based on gtsfm/runner/gtsfm_runner_base.py
        return UiMetadata(
            display_name="Image Loader",
            input_products="Source Directory",
            output_products=(
                "Images",
                "Camera Intrinsics",
                "Image Shapes",
                "Relative Pose Priors",
                "Absolute Pose Priors",
            ),
            parent_plate="Loader and Retriever",
        )

    def __init__(self, max_resolution: int = 1080, input_worker: Optional[str] = None) -> None:
        """
        Args:
            max_resolution: integer representing maximum length of image's short side
               e.g. for 1080p (1920 x 1080), max_resolution would be 1080
            input_worker: string representing ip address and the port of the worker.
        """
        if not isinstance(max_resolution, int):
            raise ValueError("Maximum image resolution must be an integer argument.")
        self._max_resolution = max_resolution
        self._input_worker = input_worker

    # ignored-abstractmethod
    @abc.abstractmethod
    def __len__(self) -> int:
        """
        The number of images in the dataset.

        Note: length should be found without loading images into memory.

        Returns:
            The number of images.
        """

    # ignored-abstractmethod
    @abc.abstractmethod
    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: The index to fetch.

        Raises:
            IndexError: If an out-of-bounds image index is requested.

        Returns:
            Image: The image at the query index.
        """

    # ignored-abstractmethod
    @abc.abstractmethod
    def get_camera_intrinsics_full_res(self, index: int) -> Optional[gtsfm_types.CALIBRATION_TYPE]:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            index: The index to fetch.

        Returns:
            Intrinsics for the given camera.
        """

    # ignored-abstractmethod
    @abc.abstractmethod
    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: The index to fetch.

        Returns:
            The camera pose w_P_index.
        """

    # ignored-abstractmethod
    @abc.abstractmethod
    def image_filenames(self) -> List[str]:
        """Return the file names corresponding to each image index."""

    # TODO: Rename this to get_gt_camera.
    def get_camera(self, index: int) -> Optional[gtsfm_types.CAMERA_TYPE]:
        """Gets the GT camera at the given index.

        Args:
            index: The index to fetch.

        Returns:
            Camera object with intrinsics and extrinsics, if they exist.
        """
        pose = self.get_camera_pose(index)
        intrinsics = self.get_gt_camera_intrinsics(index)

        if pose is None or intrinsics is None:
            return None

        camera_type = gtsfm_types.get_camera_class_for_calibration(intrinsics)

        return camera_type(pose, intrinsics)

    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair. idx1 < idx2 is required.

        Note: All inherited classes should call this super method to enforce this check.

        Args:
            idx1: First index of the pair.
            idx2: Second index of the pair.

        Returns:
            Validation result.
        """
        return idx1 < idx2

    def get_image(self, index: int) -> Image:
        """Get the image at the given index, satisfying a maximum image resolution constraint.

        Determine how the camera intrinsics and images should be jointly rescaled based on desired img. resolution.
        Each loader implementation should set a `_max_resolution` attribute.

        Args:
            index: The index to fetch.

        Raises:
            IndexError: If an out-of-bounds image index is requested.

        Returns:
            Image: The image at the query index. It will be resized to satisfy the maximum
                allowed loader image resolution if the full-resolution images for a dataset
                are too large.
        """
        # No downsampling may be required, in which case target_h and target_w will be identical
        # to the full res height & width.
        img_full_res = self.get_image_full_res(index)
        if min(img_full_res.height, img_full_res.width) <= self._max_resolution:
            return img_full_res

        # Resize image.
        (
            _,
            _,
            target_h,
            target_w,
        ) = img_utils.get_downsampling_factor_per_axis(img_full_res.height, img_full_res.width, self._max_resolution)
        logger.debug(
            "Image %d resized from (H,W)=(%d,%d) -> (%d,%d)",
            index,
            img_full_res.height,
            img_full_res.width,
            target_h,
            target_w,
        )
        resized_img = img_utils.resize_image(img_full_res, new_height=target_h, new_width=target_w)
        return resized_img

    def __rescale_intrinsics(
        self, intrinsics_full_res: gtsfm_types.CALIBRATION_TYPE, image_index: int
    ) -> gtsfm_types.CALIBRATION_TYPE:
        """Rescale the intrinsics to match the image resolution.

        Reads the image from disk to determine the scaling factor.

        Args:
            intrinsics_full_res: Intrinsics for the given camera at full resolution.
            image_index: The index to fetch.

        Returns:
            Rescaled intrinsics for the given camera at the desired resolution.
        """

        if intrinsics_full_res.fx() <= 0:
            raise RuntimeError("Focal length must be positive.")

        if intrinsics_full_res.px() <= 0 or intrinsics_full_res.py() <= 0:
            raise RuntimeError("Principal point must have positive coordinates.")

        img_full_res = self.get_image_full_res(image_index)
        # no downsampling may be required, in which case scale_u and scale_v will be 1.0
        scale_u, scale_v, _, _ = img_utils.get_downsampling_factor_per_axis(
            img_full_res.height, img_full_res.width, self._max_resolution
        )
        if isinstance(intrinsics_full_res, Cal3Bundler):
            rescaled_intrinsics = Cal3Bundler(
                fx=intrinsics_full_res.fx() * scale_u,
                k1=0.0,
                k2=0.0,
                u0=intrinsics_full_res.px() * scale_u,
                v0=intrinsics_full_res.py() * scale_v,
            )
        elif isinstance(intrinsics_full_res, Cal3_S2):
            rescaled_intrinsics = Cal3_S2(
                fx=intrinsics_full_res.fx() * scale_u,
                fy=intrinsics_full_res.fy() * scale_v,
                s=intrinsics_full_res.skew() * scale_u,
                u0=intrinsics_full_res.px() * scale_u,
                v0=intrinsics_full_res.py() * scale_v,
            )
        elif isinstance(intrinsics_full_res, Cal3DS2):
            rescaled_intrinsics = Cal3DS2(
                fx=intrinsics_full_res.fx() * scale_u,
                fy=intrinsics_full_res.fy() * scale_v,
                s=intrinsics_full_res.skew() * scale_u,
                u0=intrinsics_full_res.px() * scale_u,
                v0=intrinsics_full_res.py() * scale_v,
                k1=intrinsics_full_res.k1(),
                k2=intrinsics_full_res.k2(),
                # p1=intrinsics_full_res.p1(),  # TODO(travisdriver): figure out how to access p1 and p2
                # p2=intrinsics_full_res.p2(),
            )
        else:
            raise ValueError(f"Unsupported calibration type {type(intrinsics_full_res)} for rescaling intrinsics.")
        return rescaled_intrinsics

    def get_camera_intrinsics(self, index: int) -> Optional[gtsfm_types.CALIBRATION_TYPE]:
        """Get the camera intrinsics at the given index, for a possibly resized image.

        Determine how the camera intrinsics and images should be jointly rescaled based on desired img. resolution.
        Each loader implementation should set a `_max_resolution` attribute.

        Args:
            index: The index to fetch.

        Returns:
            Intrinsics for the given camera.
        """
        intrinsics_full_res = self.get_camera_intrinsics_full_res(index)
        if intrinsics_full_res is None:
            raise ValueError(f"No intrinsics found for index {index}.")

        return self.__rescale_intrinsics(intrinsics_full_res, index)

    def get_gt_camera_intrinsics_full_res(self, index: int) -> Optional[gtsfm_types.CALIBRATION_TYPE]:
        """Get the GT camera intrinsics at the given index, valid for a full-resolution image.

        By default, this is implemented to return the same value as `get_camera_intrinsics_full_res`. However, this can
        be overridden by subclasses to return a superior ground truth intrinsics.

        Args:
            index: The index to fetch.

        Returns:
            Intrinsics for the given camera.
        """
        return self.get_camera_intrinsics_full_res(index)

    def get_gt_camera_intrinsics(self, index: int) -> Optional[gtsfm_types.CALIBRATION_TYPE]:
        """Get the GT camera intrinsics at the given index, for a possibly resized image.

        Determine how the camera intrinsics and images should be jointly rescaled based on desired img. resolution.
        Each loader implementation should set a `_max_resolution` attribute.

        The returned intrinsics are the same as the camera intrinsics, but this behavior can be changed by overridding
        `get_gt_camera_intrinsics_full_res` in derived classes.

        Args:
            index: The index to fetch.

        Returns:
            GT intrinsics for the given camera.
        """
        intrinsics_full_res = self.get_gt_camera_intrinsics_full_res(index)
        if intrinsics_full_res is None:
            raise ValueError(f"No intrinsics found for index {index}.")

        return self.__rescale_intrinsics(intrinsics_full_res, index)

    def get_image_shape(self, idx: int) -> Tuple[int, int]:
        """Return a (H,W) tuple for each image"""
        image = self.get_image(idx)
        return (image.height, image.width)

    def get_relative_pose_prior(self, i1: int, i2: int) -> Optional[PosePrior]:
        """Get the prior on the relative pose i2Ti1

        Args:
            i1 (int): Index of first image
            i2 (int): Index of second image

        Returns:
            Pose prior, if there is one.
        """
        return None

    def get_relative_pose_priors(self, pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], PosePrior]:
        """Get *all* relative pose priors for i2Ti1

        Args:
            pairs: All (i1,i2) pairs of image pairs

        Returns:
            A dictionary of PosePriors (or None) for all pairs.
        """

        pairs = {pair: self.get_relative_pose_prior(*pair) for pair in pairs}
        return {pair: prior for pair, prior in pairs.items() if prior is not None}

    def get_absolute_pose_prior(self, idx: int) -> Optional[PosePrior]:
        """Get the prior on the pose of camera at idx in the world coordinates.

        Args:
            idx (int): Index of the camera

        Returns:
            pose prior, if there is one.
        """
        return None

    def get_absolute_pose_priors(self) -> List[Optional[PosePrior]]:
        """Get *all* absolute pose priors

        Returns:
            A list of (optional) pose priors.
        """
        N = len(self)
        return [self.get_absolute_pose_prior(i) for i in range(N)]

    def create_computation_graph_for_images(self) -> List[Delayed]:
        """Creates the computation graph for image fetches.

        Returns:
            List of delayed tasks for images.
        """
        N = len(self)
        annotation = dask.annotate(workers=self._input_worker) if self._input_worker else dask.annotate()
        with annotation:
            delayed_images = [dask.delayed(self.get_image)(i) for i in range(N)]
        return delayed_images

    def get_all_images_as_futures(self, client: Client) -> List[Future]:
        return [
            client.submit(self.get_image, i, workers=[self._input_worker] if self._input_worker else None)
            for i in range(len(self))
        ]

    def get_all_intrinsics(self) -> List[Optional[gtsfm_types.CALIBRATION_TYPE]]:
        """Return all the camera intrinsics.

        Note: use create_computation_graph_for_intrinsics when calling from runners.

        Returns:
            List of camera intrinsics.
        """
        N = len(self)
        return [self.get_camera_intrinsics(i) for i in range(N)]

    def get_gt_poses(self) -> List[Optional[Pose3]]:
        """Return all the camera poses.

        Returns:
            List of ground truth camera poses, if available.
        """
        N = len(self)
        return [self.get_camera_pose(i) for i in range(N)]

    def get_gt_cameras(self) -> List[Optional[gtsfm_types.CAMERA_TYPE]]:
        """Return all the cameras.

        Note: use create_computation_graph_for_gt_cameras when calling from runners.

        Returns:
            List of ground truth cameras, if available.
        """
        N = len(self)
        return [self.get_camera(i) for i in range(N)]

    def get_image_shapes(self) -> List[Tuple[int, int]]:
        """Return all the image shapes.

        Note: use create_computation_graph_for_image_shapes when calling from runners.

        Returns:
            List of delayed tasks for image shapes.
        """
        N = len(self)
        return [self.get_image_shape(i) for i in range(N)]

    def get_valid_pairs(self) -> List[Tuple[int, int]]:
        """Get the valid pairs of images for this loader.

        Returns:
            List of valid index pairs.
        """
        pairs = []

        for idx1 in range(self.__len__()):
            for idx2 in range(self.__len__()):
                if self.is_valid_pair(idx1, idx2):
                    pairs.append((idx1, idx2))

        return pairs

    def get_gt_scene_trimesh(self) -> Optional[Trimesh]:
        """Getter for the ground truth mesh for the scene.

        Returns:
            Trimesh object, if available
        """
        return None

    def get_images_with_exif(self, search_path: str) -> Tuple[List[str], int]:
        """Return images with exif.
        Args:
            search_path: image sequence search path.
        Returns:
            Tuple[
                List of image with exif paths.
                The number of all the images.
            ]
        """
        all_image_paths = io_utils.get_sorted_image_names_in_dir(search_path)
        num_all_imgs = len(all_image_paths)
        exif_image_paths = []
        for single_img_path in all_image_paths:
            # Drop images without exif.
            if io_utils.load_image(single_img_path).get_intrinsics_from_exif() is None:
                continue
            exif_image_paths.append(single_img_path)

        return (exif_image_paths, num_all_imgs)
