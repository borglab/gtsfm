"""Simple loader class that reads a dataset with metadata formatted in the COLMAP style.

Authors: John Lambert
"""

import os
from pathlib import Path
from typing import List, Optional

from gtsam import Cal3Bundler, Pose3  # type:ignore

import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase

logger = logger_utils.get_logger()


class ColmapLoader(LoaderBase):
    """Simple loader class that reads a dataset with ground-truth files and dataset meta-information
    formatted in the COLMAP style. This meta-information may include image file names stored
    in a images.txt file, or ground truth camera poses for each image/frame. Images should be
    present in the specified image directory.

    Note that these may not be actual ground truth, but just pseudo-ground truth, i.e. output of
    COLMAP.

    Note: assumes all images are of the same dimensions.
    TODO: support max-resolution constraint for images are of the same dimensions.
       Would introduce a constraint between get_image() and get_intrinsics()

    Folder layout structure:
    - RGB Images: images/
    - Intrinsics data (optional): cameras.txt (see https://colmap.github.io/format.html#cameras-txt)
    - Extrinsics data (optional): images.txt (see https://colmap.github.io/format.html#images-txt)

    If explicit intrinsics are not provided, the exif data will be used.
    """

    def __init__(
        self,
        dataset_dir: str,
        images_dir: Optional[str] = None,
        colmap_files_subdir: Optional[str] = None,
        use_gt_intrinsics: bool = True,
        use_gt_extrinsics: bool = True,
        max_resolution: int = 760,
        input_worker: Optional[str] = None,
    ) -> None:
        """Initializes to load from a specified dataset directory on disk.

        Args:
            dataset_dir: Path to the dataset root directory. COLMAP-exported text files
                (images.txt, cameras.txt) are expected either directly under this directory
                or under ``colmap_files_subdir`` if provided.
            images_dir: Path to directory containing image files. If None, defaults to
                {dataset_dir}/images.
            colmap_files_subdir: Optional subdirectory under dataset_dir where COLMAP files
                are located (e.g. sparse/0). If None, COLMAP files are read from dataset_dir.
            use_gt_intrinsics: Whether to use ground truth intrinsics. If COLMAP calibration is
               not found on disk, then use_gt_intrinsics will be set to false automatically.
            use_gt_extrinsics: Whether to use ground truth extrinsics.
            max_resolution: Integer representing maximum length of image's short side, i.e.
               the smaller of the height/width of the image. e.g. for 1080p (1920 x 1080),
               max_resolution would be 1080. If the image resolution max(height, width) is
               greater than the max_resolution, it will be downsampled to match the max_resolution.
        """
        super().__init__(max_resolution, input_worker)
        self._dataset_dir = dataset_dir
        self._images_dir = images_dir or os.path.join(dataset_dir, "images")
        self._colmap_files_subdir = colmap_files_subdir
        self._use_gt_intrinsics = use_gt_intrinsics
        self._use_gt_extrinsics = use_gt_extrinsics
        colmap_data_dir = (
            os.path.join(self._dataset_dir, self._colmap_files_subdir)
            if self._colmap_files_subdir is not None
            else self._dataset_dir
        )

        wTi_list, img_fnames, self._calibrations, _, _, _ = io_utils.read_scene_data_from_colmap_format(colmap_data_dir)
        # TODO in future PR: if img_fnames is None, default to using everything inside image directory

        if self._calibrations is None:
            self._use_gt_intrinsics = False

        if self._calibrations is not None and len(self._calibrations) == 1:
            # shared calibration!
            self._calibrations = self._calibrations * len(img_fnames)

        # Preserve COLMAP ordering of images.

        self._img_fnames = []
        self._image_paths = []
        self._wTi_list = []

        # If one of the images is not found on disk, the assigned image indices will be re-ordered on disk
        # to skip the missing image.
        for img_fname, wTi in zip(img_fnames, wTi_list):
            img_fpath = os.path.join(self._images_dir, img_fname)
            if not Path(img_fpath).exists():
                continue
            self._img_fnames.append(img_fname)
            self._image_paths.append(img_fpath)
            self._wTi_list.append(wTi)

        self._num_imgs = len(self._image_paths)
        logger.info("Colmap image loader found and loaded %d images", self._num_imgs)

    def image_filenames(self) -> List[str]:
        """Return the file names corresponding to each image index."""
        return self._img_fnames

    def get_image_fname(self, idx: int) -> str:
        """Given an image index, provide the corresponding image filename."""
        return Path(self._image_paths[idx]).name

    def get_image_index_from_filename(self, fname: str) -> int:
        """Given an image filename, provide the corresponding image index."""
        return self._img_fnames.index(fname)

    def __len__(self) -> int:
        """The number of images in the dataset.

        Returns:
            The number of images.
        """
        return self._num_imgs

    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: The index to fetch.

        Returns:
            Image: The image at the query index.

        Raises:
            IndexError: If an out-of-bounds image index is requested.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")

        img = io_utils.load_image(self._image_paths[index])
        return img

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3Bundler]:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            The index to fetch.

        Returns:
            Intrinsics for the given camera.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid. Valid indices are in [0,{len(self)-1}].")

        if not self._use_gt_intrinsics:
            # get intrinsics from exif
            intrinsics = io_utils.load_image(self._image_paths[index]).get_intrinsics_from_exif()
        else:
            intrinsics = self._calibrations[index]

        return intrinsics

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: The index to fetch.

        Returns:
            The camera pose w_T_index.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid. Valid indices are in [0,{len(self)-1}].")

        if not self._use_gt_extrinsics:
            return None

        wTi = self._wTi_list[index]
        return wTi
