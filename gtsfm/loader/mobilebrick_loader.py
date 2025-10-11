"""Simple loader class that reads from the MobileBrick dataset.

Reference to MobileBrick: https://code.active.vision/MobileBrick/, Kejie Li et al.

Authors: Akshay Krishnan
"""

import os
from typing import List, Optional

import numpy as np
from gtsam import Cal3Bundler, Pose3, Rot3  # type:ignore

import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase

logger = logger_utils.get_logger()


class MobilebrickLoader(LoaderBase):
    """Loader class that reads from the MobileBrick dataset."""

    def __init__(
        self,
        dataset_dir: str,
        images_dir: Optional[str] = None,
        use_gt_intrinsics: bool = False,
        max_frame_lookahead: int = 5,
        max_resolution: int = 1024,
        input_worker: Optional[str] = None,
    ) -> None:
        """Initialize loader for MobileBrick dataset.

        Args:
            dataset_dir: Path to dataset directory.
            images_dir: Path to images directory. If None, defaults to {dataset_dir}/image.
            use_gt_intrinsics: Whether to use ground truth intrinsics.
            max_frame_lookahead: Maximum frame lookahead for matching.
            max_resolution: Maximum image resolution.
            input_worker: Optional input worker specification.
        """
        super().__init__(max_resolution=max_resolution, input_worker=input_worker)

        self._dataset_dir = dataset_dir
        self._images_dir = images_dir or os.path.join(dataset_dir, "image")
        self._use_gt_intrinsics = use_gt_intrinsics
        self._max_frame_lookahead = max_frame_lookahead
        self._num_images = len(os.listdir(self._images_dir))

        # Cache image paths
        self._image_paths = []
        for i in range(self._num_images):
            image_path = os.path.join(self._images_dir, f"{i:06d}.jpg")
            self._image_paths.append(image_path)

        # Load GT intrinsics
        intrinsics_dir = os.path.join(self._dataset_dir, "intrinsic")
        self._gt_intrinsics = []
        for i in range(self._num_images):
            intrinsics_file = os.path.join(intrinsics_dir, f"{i:06d}.txt")
            K = np.loadtxt(intrinsics_file)
            self._gt_intrinsics.append(Cal3Bundler((K[0, 0] + K[1, 1]) / 2, 0, 0, K[0, 2], K[1, 2]))

        # Load GT poses
        self._poses_dir = os.path.join(self._dataset_dir, "pose")
        self._wTi = []
        for i in range(self._num_images):
            pose_file = os.path.join(self._poses_dir, f"{i:06d}.txt")
            wTi_mat = np.loadtxt(pose_file)
            wTi = Pose3(Rot3(wTi_mat[:3, :3]), wTi_mat[:3, 3])  # type: ignore
            self._wTi.append(wTi)

    def image_filenames(self) -> List[str]:
        """Return the file names corresponding to each image index."""
        return [fpath for fpath in sorted(os.listdir(self._images_dir))]

    def __len__(self) -> int:
        """The number of images in the dataset.

        Returns:
            The number of images.
        """
        return self._num_images

    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: The index to fetch.

        Returns:
            The image at the query index.

        Raises:
            IndexError: If an out-of-bounds image index is requested.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")

        # Read in image.
        img = io_utils.load_image(self._image_paths[index])
        return Image(value_array=img.value_array, exif_data=img.exif_data, file_name=img.file_name)

    def get_gt_camera_intrinsics_full_res(self, index: int) -> Cal3Bundler:
        """Get the GT camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            index: The index to fetch.

        Returns:
            Ground truth intrinsics for the given camera.

        Raises:
            IndexError: If an out-of-bounds image index is requested.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")

        return self._gt_intrinsics[index]

    def get_camera_intrinsics_full_res(self, index: int) -> Cal3Bundler:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            index: The index to fetch.

        Returns:
            Ground truth intrinsics for the given camera.

        Raises:
            IndexError: If an out-of-bounds image index is requested.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")

        if self._use_gt_intrinsics:
            return self._gt_intrinsics[index]
        else:
            # 0.8 is better than the default factor of 1.2 for this dataset, but it has not been fully tuned.
            image = io_utils.load_image(self._image_paths[index])
            assert image is not None, "Image could not be read."
            calibration = image.get_intrinsics(default_focal_length_factor=0.8)
            assert calibration is not None, "Calibration could not be retrieved."
            return calibration

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: The index to fetch.

        Returns:
            Ground truth pose for the given camera.

        Raises:
            IndexError: If an out-of-bounds image index is requested.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")

        wTi = self._wTi[index]
        return wTi

    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair. idx1 < idx2 is required.

        Args:
            idx1: First index of the pair.
            idx2: Second index of the pair.

        Returns:
            Validation result.
        """
        return super().is_valid_pair(idx1, idx2) and abs(idx1 - idx2) <= self._max_frame_lookahead
