"""Base class for Loaders.

Authors: Frank Dellaert and Ayush Baid
"""

import abc
from typing import Callable, Dict, List, Optional, Tuple, TypeAlias, Union

import numpy as np
import torch
from dask.base import annotate as dask_annotate
from dask.delayed import Delayed, delayed
from dask.distributed import Client, Future
from gtsam import Cal3_S2, Cal3Bundler, Cal3DS2, Pose3  # type: ignore
from PIL import Image as PILImage
from torchvision import transforms as TF
from trimesh import Trimesh

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.images as img_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.common.pose_prior import PosePrior
from gtsfm.products.one_view_data import OneViewData
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata

ArrayLike = Union[np.ndarray, torch.Tensor]
ResizeTransform: TypeAlias = Callable[[ArrayLike], torch.Tensor]
BatchTransform: TypeAlias = Callable[[torch.Tensor], torch.Tensor]

logger = logger_utils.get_logger()


class LoaderBase(GTSFMProcess):
    """Base class for Loaders.

    The loader provides APIs to get an image, either directly or as a dask delayed task
    """

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        # based on gtsfm/runner.py
        return UiMetadata(
            display_name="Image Loader",
            input_products=("Source Directory",),
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

        return camera_type(pose, intrinsics)  # type: ignore

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

        The returned intrinsics are the same as the camera intrinsics, but this behavior can be changed by overriding
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

    def get_relative_pose_priors(self, visibility_graph: VisibilityGraph) -> Dict[Tuple[int, int], PosePrior]:
        """Get *all* relative pose priors for i2Ti1

        Args:
            visibility_graph: The visibility graph defining which image pairs to get priors for

        Returns:
            A dictionary of PosePriors (or None) for all pairs.
        """

        pairs = {pair: self.get_relative_pose_prior(*pair) for pair in visibility_graph}
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

    def get_image_futures(self, client: Client) -> Dict[int, Future]:
        """Submit one get_image future per index and return them keyed by index.

        Args:
            client: Dask client responsible for executing the image load tasks.

        Returns:
            Dictionary mapping image index -> Future resolving to `Image`.
        """
        workers = [self._input_worker] if self._input_worker else None
        future_map: Dict[int, Future] = {}
        for idx in range(len(self)):
            future_map[idx] = client.submit(
                self.get_image,
                idx,
                workers=workers,
                key=f"loader-get-image-{idx}",
            )
        return future_map

    def get_key_images_as_delayed_map(self, keys: List[int]) -> Dict[int, Delayed]:
        """Creates a computation graph to fetch images, using the provided keys as identifiers."""

        annotation = dask_annotate(workers=self._input_worker) if self._input_worker else dask_annotate()
        with annotation:
            delayed_images = {key: delayed(self.get_image)(key) for key in keys}
        return delayed_images

    @staticmethod
    # do padding to square and resize to target size
    def pad_image(img: np.ndarray, max_side: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Pad image to square dimensions using the maximum side length observed in the batch."""
        pad_height = max_side - img.shape[0]
        pad_width = max_side - img.shape[1]

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padded = np.pad(
            img,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return padded, (pad_top, pad_bottom, pad_left, pad_right)

    def load_image_batch(
        self,
        indices: List[int],
        resize_transform: ResizeTransform,
        batch_transform: Optional[BatchTransform] = None,
    ) -> torch.Tensor:
        """Helper function that runs on a Dask worker to load a batch of images.

        Args:
            indices: List of image indices to load
            resize_transform: callable that converts a numpy array (image) to a torch.Tensor of the desired size.
            batch_transform: Optional callable that applies a preprocessing transform to the batch tensor.

        Returns:
            torch.Tensor: Batch of loaded (and optionally transformed) images.
        """
        # Get images as a List of [H, W, C] numpy arrays
        image_arrays = [self.get_image(idx).value_array for idx in indices]

        # Determine whether all images share the same spatial size.
        base_shape = image_arrays[0].shape[:2]
        shapes_match = all(img.shape[:2] == base_shape for img in image_arrays)

        if shapes_match:
            working_arrays = image_arrays
        else:
            # Pad each image to square dimensions using the maximum side length observed in the batch.
            max_side = max(max(img.shape[0], img.shape[1]) for img in image_arrays)
            working_arrays = []
            for img in image_arrays:
                padded, _ = self.pad_image(img, max_side)
                working_arrays.append(padded)

        image_tensors = [resize_transform(arr) for arr in working_arrays]
        batch_tensor = torch.stack(image_tensors, dim=0)

        # Apply optional batch transform before returning
        return batch_transform(batch_tensor) if batch_transform else batch_tensor

    def load_image_batch_vggt(
        self,
        indices: List[int],
        img_load_resolution: int,
        resize_transform: ResizeTransform,
        batch_transform: Optional[BatchTransform] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper function that runs on a Dask worker to load a batch of images.

        Args:
            indices: List of image indices to load
            img_load_resolution: target image resolution for loading images.
            resize_transform: callable that converts a numpy array (image) to a torch.Tensor of the desired size.
            batch_transform: Optional callable that applies a preprocessing transform to the batch tensor.

        Returns:
            torch.Tensor: Batch of loaded (and optionally transformed) images.
        """
        # Get images as a List of [H, W, C] numpy arrays
        image_arrays = [self.get_image(idx).value_array for idx in indices]

        working_arrays = []
        original_coords = []
        for img in image_arrays:
            max_side = max(img.shape[0], img.shape[1])
            padded, (pad_top, _, pad_left, _) = self.pad_image(img, max_side)
            working_arrays.append(padded)

            scale = img_load_resolution / max_side
            x1 = pad_left * scale
            y1 = pad_top * scale
            x2 = (pad_left + img.shape[1]) * scale
            y2 = (pad_top + img.shape[0]) * scale

            original_coords.append(np.array([x1, y1, x2, y2, img.shape[1], img.shape[0]]))

        image_tensors = [resize_transform(arr) for arr in working_arrays]
        batch_tensor = torch.stack(image_tensors, dim=0)

        original_coords_tensor = torch.from_numpy(np.array(original_coords)).float()

        # Apply optional batch transform before returning
        transformed = batch_transform(batch_tensor) if batch_transform else batch_tensor
        return transformed, original_coords_tensor

    def load_image_batch_vggt_loader(self, indices: List[int], mode="crop"):
        """
        A quick start function to load and preprocess images for model input.
        This assumes the images should have the same shape for easier batching,
        but VGGT model can also work well with different shapes.

        Args:
            indices: List of image indices to load
            mode (str, optional): Preprocessing mode, either "crop" or "pad".
                                - "crop" (default): Sets width to 518px and center crops height if needed.
                                - "pad": Preserves all pixels by making the largest dimension 518px
                                and padding the smaller dimension to reach a square shape.

        Returns:
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

        Raises:
            ValueError: If the input list is empty or if mode is invalid

        Notes:
            - Images with different dimensions will be padded with white (value=1.0)
            - A warning is printed when images have different shapes
            - When mode="crop": The function ensures width=518px while maintaining aspect ratio
            and height is center-cropped if larger than 518px
            - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
            and the smaller dimension is padded to reach a square shape (518x518)
            - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
        """
        # Check for empty list
        if len(indices) == 0:
            raise ValueError("At least 1 image is required")

        # Validate mode
        if mode not in ["crop", "pad"]:
            raise ValueError("Mode must be either 'crop' or 'pad'")

        images = []
        shapes = set()
        to_tensor = TF.ToTensor()
        target_size = 518

        # First process all images and collect their shapes
        for idx in indices:
            # Open image
            img = self.get_image(idx).value_array

            img = PILImage.fromarray(img)

            width, height = img.size

            if mode == "pad":
                # Make the largest dimension 518px while maintaining aspect ratio
                if width >= height:
                    new_width = target_size
                    new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
                else:
                    new_height = target_size
                    new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
            else:  # mode == "crop"
                # Original behavior: set width to 518px
                new_width = target_size
                # Calculate height maintaining aspect ratio, divisible by 14
                new_height = round(height * (new_width / width) / 14) * 14

            # Resize with new dimensions (width, height)
            img = img.resize((new_width, new_height), PILImage.Resampling.BICUBIC)
            img = to_tensor(img)  # Convert to tensor (0, 1)

            # Center crop height if it's larger than 518 (only in crop mode)
            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                img = img[:, start_y : start_y + target_size, :]

            # For pad mode, pad to make a square of target_size x target_size
            if mode == "pad":
                h_padding = target_size - img.shape[1]
                w_padding = target_size - img.shape[2]

                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left

                    # Pad with white (value=1.0)
                    img = torch.nn.functional.pad(
                        img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                    )

            shapes.add((img.shape[1], img.shape[2]))
            images.append(img)

        # Check if we have different shapes
        # In theory our model can also work well with different shapes
        if len(shapes) > 1:
            logger.warning("Found images with different shapes: %s", shapes)
            # Find maximum dimensions
            max_height = max(shape[0] for shape in shapes)
            max_width = max(shape[1] for shape in shapes)

            # Pad images if necessary
            padded_images = []
            for img in images:
                h_padding = max_height - img.shape[1]
                w_padding = max_width - img.shape[2]

                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left

                    img = torch.nn.functional.pad(
                        img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                    )
                padded_images.append(img)
            images = padded_images

        images = torch.stack(images)  # concatenate images

        # Ensure correct shape when single image
        if len(indices) == 1:
            # Verify shape is (1, C, H, W)
            if images.dim() == 3:
                images = images.unsqueeze(0)

        height, width = images.shape[-2], images.shape[-1]
        coords = np.tile([0.0, 0.0, float(width), float(height), float(width), float(height)], (len(indices), 1))
        original_coords_tensor = torch.from_numpy(coords).float()

        return images, original_coords_tensor

    def get_all_descriptor_image_batches_as_futures(
        self,
        client: Client,
        batch_size: int,
        resize_transform: ResizeTransform,
        batch_transform: Optional[BatchTransform] = None,
    ) -> List[Future]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        workers = [self._input_worker] if self._input_worker else None
        num_images = len(self)

        index_batches = [
            list(range(start, min(start + batch_size, num_images))) for start in range(0, num_images, batch_size)
        ]

        batch_futures = [
            client.submit(self.load_image_batch, indices, resize_transform, batch_transform, workers=workers)
            for indices in index_batches
        ]

        return batch_futures

    def get_all_intrinsics(self) -> List[Optional[gtsfm_types.CALIBRATION_TYPE]]:
        """Return all the camera intrinsics.

        Note: use create_computation_graph_for_intrinsics when calling from runners.

        Returns:
            List of camera intrinsics.
        """
        N = len(self)
        return [self.get_camera_intrinsics(i) for i in range(N)]

    def get_one_view_data_dict(self) -> Dict[int, OneViewData]:
        """Construct a per-view data map keyed by image index along with validated intrinsics.

        Returns:
            Dictionary mapping image index to OneViewData with eagerly validated intrinsics.
        """
        maybe_intrinsics = self.get_all_intrinsics()
        if any(intrinsic is None for intrinsic in maybe_intrinsics):
            raise ValueError("Some intrinsics are None. Please ensure all intrinsics are provided.")

        intrinsics: List[gtsfm_types.CALIBRATION_TYPE] = maybe_intrinsics  # type: ignore
        image_fnames = self.image_filenames()
        absolute_pose_priors = self.get_absolute_pose_priors()
        cameras_gt = self.get_gt_cameras()
        gt_wTi_list = self.get_gt_poses()

        num_images = len(self)
        if not (
            len(maybe_intrinsics)
            == len(image_fnames)
            == len(absolute_pose_priors)
            == len(cameras_gt)
            == len(gt_wTi_list)
            == num_images
        ):
            raise ValueError("Per-view inputs must match the number of images in the loader.")

        one_view_data_dict = {
            idx: OneViewData(
                image_fname=image_fnames[idx],
                intrinsics=intrinsics[idx],
                absolute_pose_prior=absolute_pose_priors[idx],
                camera_gt=cameras_gt[idx],
                pose_gt=gt_wTi_list[idx],
            )
            for idx in range(num_images)
        }
        return one_view_data_dict

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

    def get_valid_pairs(self) -> VisibilityGraph:
        """Get the valid pairs of images for this loader.

        Returns:
            Visibility graph of valid index pairs.
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
