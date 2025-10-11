"""Simple loader class that reads a dataset with metadata formatted in the COLMAP style.

Authors: Travis Driver
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np
import trimesh
from gtsam import Pose3, SfmTrack  # type:ignore
from trimesh import Trimesh

import gtsfm.utils.images as image_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import thirdparty.colmap.scripts.python.read_write_model as colmap_io
from gtsfm.common.image import Image
from gtsfm.common.types import CALIBRATION_TYPE
from gtsfm.loader.loader_base import LoaderBase

logger = logger_utils.get_logger()


class AstrovisionLoader(LoaderBase):
    """Loader class that reads an AstroVision data segment.

    Refs:
    - https://github.com/astrovision
    """

    def __init__(
        self,
        dataset_dir: str,
        images_dir: Optional[str] = None,
        gt_scene_mesh_path: str | None = None,
        use_gt_extrinsics: bool = True,
        use_gt_sfm_tracks: bool = False,
        use_gt_masks: bool = False,
        max_frame_lookahead: int = 2,
        max_resolution: int = 1024,
    ) -> None:
        """Initialize loader from a specified dataset directory on disk.

        <dataset_dir>/
             ├── images/: undistorted grayscale images
             ├── cameras.bin: camera calibrations (see https://colmap.github.io/format.html#cameras-txt)
             ├── images.bin: 3D poses and 2D tracks (see https://colmap.github.io/format.html#images-txt)
             └── points3D.bin: 3D tracks (see https://colmap.github.io/format.html#points3d-txt)

        Args:
            dataset_dir: Path to directory containing the COLMAP-formatted data: cameras.bin,
                images.bin, and points3D.bin
            images_dir: Path to images directory. If None, defaults to {dataset_dir}/images.
            gt_scene_mesh_path (optional): Path to file of target small body surface mesh.
                Note: vertex size mismatch observed when reading in from OBJ format. Prefer PLY.
            use_gt_extrinsics (optional): Whether to use ground truth extrinsics. Used only for comparison with
                reconstructed values.
            use_gt_sfm_tracks (optional): Whether to use ground truth tracks. Used only for comparison w reconstructed
                values.
            use_masks (optional): Whether to use ground truth masks.
            max_frame_lookahead (optional): Maximum number of consecutive frames to consider for
                matching/co-visibility. Any value of max_frame_lookahead less than the size of
                the dataset assumes data is sequentially captured.
            max_resolution: Integer representing maximum length of image's short side, i.e.
               the smaller of the height/width of the image. e.g. for 1080p (1920 x 1080),
               max_resolution would be 1080. If the image resolution max(height, width) is
               greater than the max_resolution, it will be downsampled to match the max_resolution.

        Raises:
            FileNotFoundError: If `dataset_dir` doesn't exist or image path does not exist.
            RuntimeError: If ground truth camera calibrations not provided.
        """
        super().__init__(max_resolution)
        self._dataset_dir = dataset_dir
        self._images_dir = images_dir or os.path.join(dataset_dir, "images")
        self._use_gt_extrinsics = use_gt_extrinsics
        self._use_gt_sfm_tracks = use_gt_sfm_tracks
        self._max_frame_lookahead = max_frame_lookahead

        # Use COLMAP model reader to load data and convert to GTSfM format.
        if not Path(dataset_dir).exists():
            raise FileNotFoundError("No data found at %s." % dataset_dir)
        cameras, images, points3d = colmap_io.read_model(path=dataset_dir, ext=".bin")

        img_fnames, self._wTi_list, self._calibrations, self._sfm_tracks, _, _, _ = io_utils.colmap2gtsfm(
            cameras, images, points3d, load_sfm_tracks=use_gt_sfm_tracks
        )

        # Read in scene mesh as Trimesh object.
        if gt_scene_mesh_path is not None:
            if not Path(gt_scene_mesh_path).exists():
                raise FileNotFoundError(f"No mesh found at {gt_scene_mesh_path}")
            self._gt_scene_trimesh = trimesh.load(gt_scene_mesh_path, process=False, maintain_order=True)
            logger.info(
                "AstroVision loader read in mesh with %d vertices and %d faces.",
                self._gt_scene_trimesh.vertices.shape[0],
                self._gt_scene_trimesh.faces.shape[0],
            )
        else:
            self._gt_scene_trimesh = None

        # Camera intrinsics are currently required due to absence of EXIF data and difficulty in approximating focal
        # length (usually 10000 to 100000 pixels).
        if self._calibrations is None:
            raise RuntimeError("Camera intrinsics cannot be None.")

        if self._wTi_list is None and self._use_gt_extrinsics:
            raise RuntimeError("Ground truth extrinsic data requested but missing.")

        if self._sfm_tracks is None and self._use_gt_sfm_tracks:
            raise RuntimeError("Ground truth SfMTrack data requested but missing.")
        self.num_sfm_tracks = len(self._sfm_tracks) if self._sfm_tracks is not None else 0

        # Prepare image paths.
        self._image_paths: List[str] = []
        self._mask_paths: Optional[List[str]] = [] if use_gt_masks else None
        for img_fname in img_fnames:
            img_fpath = os.path.join(self._images_dir, img_fname)
            if not Path(img_fpath).exists():
                raise FileNotFoundError(f"Could not locate image at {img_fpath}.")
            self._image_paths.append(img_fpath)
            if use_gt_masks and self._mask_paths is not None:  # None check to appease mypy
                self._mask_paths.append(os.path.join(self._dataset_dir, "masks", img_fname))

        self._num_imgs = len(self._image_paths)
        logger.info("AstroVision loader found and loaded %d images and %d tracks.", self._num_imgs, self.num_sfm_tracks)

    def image_filenames(self) -> List[str]:
        """Return the file names corresponding to each image index."""
        return [str(Path(fpath)) for fpath in self._image_paths]

    def __len__(self) -> int:
        """The number of images in the dataset.

        Returns:
            the number of images.
        """
        return self._num_imgs

    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: The index to fetch.

        Raises:
            IndexError: If an out-of-bounds image index is requested.

        Returns:
            Image: The image at the query index.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")

        # Read in image.
        img = io_utils.load_image(self._image_paths[index])

        # Generate mask to separate background deep space from foreground target body
        # based on image intensity values.
        if self._mask_paths is not None:  # feed-forward masks
            mask = image_utils.rgb_to_gray_cv(io_utils.load_image(self._mask_paths[index])).value_array
            mask[mask > 0] = 1
        else:
            mask = get_nonzero_intensity_mask(img)

        return Image(value_array=img.value_array, exif_data=img.exif_data, file_name=img.file_name, mask=mask)

    def get_camera_intrinsics_full_res(self, index: int) -> CALIBRATION_TYPE:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            index: The index to fetch.

        Returns:
            Intrinsics for the given camera.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")
        intrinsics = self._calibrations[index]
        logger.debug("Loading ground truth calibration.")

        return intrinsics

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            pose for the given camera.
        """
        if not self._use_gt_extrinsics or self._wTi_list is None:
            return None

        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")

        wTi = self._wTi_list[index]
        return wTi

    def get_sfm_track(self, index: int) -> Optional[SfmTrack]:
        """Get the SfmTracks(s) (in world coordinates) at the given index.

        Args:
            index: The index to fetch.

        Returns:
            SfmTrack at index.
        """
        if not self._use_gt_sfm_tracks or self._sfm_tracks is None:
            return None

        if index < 0 or index >= len(self._sfm_tracks):
            raise IndexError(f"Track3D index {index} is invalid")

        return self._sfm_tracks[index]

    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair. idx1 < idx2 is required.

        Args:
            idx1: First index of the pair.
            idx2: Second index of the pair.

        Returns:
            Validation result.
        """
        return super().is_valid_pair(idx1, idx2) and abs(idx1 - idx2) <= self._max_frame_lookahead

    def get_gt_scene_trimesh(self) -> Optional[Trimesh]:
        """Getter for the ground truth mesh for the scene.

        Returns:
            Trimesh object, if available
        """
        return self._gt_scene_trimesh


def get_nonzero_intensity_mask(img: Image, eps: int = 5, kernel_size: Tuple[int, int] = (15, 15)) -> np.ndarray:
    """Generate mask of where image intensity values are non-zero.

    After thresholding the image, we use an erosion kernel to add a buffer between the foreground and background.

    Args:
        img: Input Image to be masked (values in range [0, 255]).
        eps: Minimum allowable intensity value, i.e., values below this value will be masked out.
        kernel_size: Size of erosion kernel.

    Returns:
        Mask (as an integer array) of Image where with a value of 1 where the intensity value is above `eps` and 0
        otherwise.
    """
    gray_image = image_utils.rgb_to_gray_cv(img)
    _, binary_image = cv.threshold(gray_image.value_array, eps, 255, cv.THRESH_BINARY)
    mask = cv.erode(binary_image, np.ones(kernel_size, np.uint8)) // 255

    return mask
