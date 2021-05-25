"""Simple loader class that reads a dataset with metadata formatted in the COLMAP style.

Authors: John Lambert
"""

import os
from pathlib import Path
from typing import Optional

from gtsam import Cal3Bundler, Pose3

import gtsfm.utils.images as img_utils
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
        colmap_files_dirpath: str,
        images_dir: str,
        use_gt_intrinsics: bool = True,
        use_gt_extrinsics: bool = True,
        max_frame_lookahead: int = 1,
        max_resolution: int = 760,
    ) -> None:
        """Initializes to load from a specified folder on disk.

        Args:
            colmap_files_dirpath: path to directory containing COLMAP-exported data, with images.txt
                and cameras.txt files
            images_dir: path to directory containing images files
            use_gt_intrinsics: whether to use ground truth intrinsics. If COLMAP calibration is
               not found on disk, then use_gt_intrinsics will be set to false automatically.
            use_gt_extrinsics: whether to use ground truth extrinsics
            max_frame_lookahead: maximum number of consecutive frames to consider for
                matching/co-visibility. Any value of max_frame_lookahead less than the size of
                the dataset assumes data is sequentially captured
            max_resolution: integer representing maximum length of image's short side
               e.g. for 1080p (1920 x 1080), max_resolution would be 1080
        """
        self._use_gt_intrinsics = use_gt_intrinsics
        self._use_gt_extrinsics = use_gt_extrinsics
        self._max_frame_lookahead = max_frame_lookahead
        self._max_resolution = max_resolution

        self._wTi_list, img_fnames = io_utils.read_images_txt(fpath=os.path.join(colmap_files_dirpath, "images.txt"))
        self._calibrations = io_utils.read_cameras_txt(fpath=os.path.join(colmap_files_dirpath, "cameras.txt"))

        # TODO in future PR: if img_fnames is None, default to using everything inside image directory

        if self._calibrations is None:
            self._use_gt_intrinsics = False

        if self._calibrations is not None and len(self._calibrations) == 1:
            # shared calibration!
            self._calibrations = self._calibrations * len(img_fnames)

        # preserve COLMAP ordering of images
        self._image_paths = []
        for img_fname in img_fnames:
            img_fpath = os.path.join(images_dir, img_fname)
            if not Path(img_fpath).exists():
                continue
            self._image_paths.append(img_fpath)

        self._num_imgs = len(self._image_paths)
        logger.info("Colmap image loader found and loaded %d images", self._num_imgs)

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


    def __len__(self) -> int:
        """The number of images in the dataset.

        Returns:
            the number of images.
        """
        return self._num_imgs

    def get_image(self, index: int) -> Image:
        """Get the image at the given index.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            Image: the image at the query index.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Image index is invalid")

        img = io_utils.load_image(self._image_paths[index])
        img = img_utils.resize_image(img, new_height=self._target_h, new_width=self._target_w)
        return img

    def get_camera_intrinsics(self, index: int) -> Cal3Bundler:
        """Get the camera intrinsics at the given index.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Image index is invalid")

        if not self._use_gt_intrinsics:
            # get intrinsics from exif
            intrinsics = io_utils.load_image(self._image_paths[index]).get_intrinsics_from_exif()

        else:
            intrinsics = self._calibrations[index]

        intrinsics = Cal3Bundler(
            fx=intrinsics.fx() * self._scale_u,
            k1=0.0,
            k2=0.0,
            u0=intrinsics.px() * self._scale_u,
            v0=intrinsics.py() * self._scale_v,
        )
        return intrinsics

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            the camera pose w_T_index.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Image index is invalid")

        if not self._use_gt_extrinsics:
            return None

        wTi = self._wTi_list[index]
        return wTi

    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """
        return idx1 < idx2 and abs(idx1 - idx2) <= self._max_frame_lookahead
