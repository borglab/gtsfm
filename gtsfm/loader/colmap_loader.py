"""Simple loader class that reads a dataset with metadata formatted in the COLMAP style.

Authors: John Lambert
"""

import os
from typing import Optional

from gtsam import Cal3Bundler, Pose3

import gtsfm.utils.io as io_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase


class ColmapLoader(LoaderBase):
    """Simple loader class that reads a dataset with metadata formatted in the COLMAP style.

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
        max_frame_lookahead: int = 1
    ) -> None:
        """Initializes to load from a specified folder on disk.

        Args:
            colmap_files_dirpath: path to directory containing COLMAP-exported data, with images.txt
                and cameras.txt files
            images_dir: path to directory containing images files
            use_gt_intrinsics: whether to use ground truth intrinsics
            use_gt_extrinsics: whether to use ground truth extrinsics
            max_frame_lookahead: if images were sequentially captured, maximum number
               of frames to consider for matching/co-visibility. Defaults to 1, i.e. 
               assuming data is not sequentially captured.
        """
        self._use_gt_intrinsics = use_gt_intrinsics
        self._use_gt_extrinsics = use_gt_extrinsics
        self._max_frame_lookahead = max_frame_lookahead

        self._calibrations = io_utils.read_cameras_txt(fpath=os.path.join(colmap_files_dirpath,"cameras.txt"))
        self._wTi_list, img_fnames = io_utils.read_images_txt(fpath=os.path.join(colmap_files_dirpath, "images.txt"))

        if self._calibrations is None or len(img_fnames) != len(self._calibrations):
            self._use_gt_intrinsics = False

        # preserve COLMAP ordering of images
        self._image_paths = []
        for img_fname in img_fnames:
            img_fpath = os.path.join(images_dir, img_fname)
            self._image_paths.append(img_fpath)

        self._num_imgs = len(self._image_paths)

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

        if index < 0 or index > self.__len__():
            raise IndexError("Image index is invalid")

        return io_utils.load_image(self._image_paths[index])


    def get_camera_intrinsics(self, index: int) -> Cal3Bundler:
        """Get the camera intrinsics at the given index.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        if not self._use_gt_intrinsics:
            # get intrinsics from exif
            intrinsics = io_utils.load_image(self._image_paths[index]).get_intrinsics_from_exif()

        else:
            intrinsics = self._calibrations[index]
        return intrinsics


    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            the camera pose w_T_index.
        """
        if not self._use_gt_extrinsics:
            return None

        wTi = self._wTi_list[index]
        return wTi


    def validate_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """
        return idx1 < idx2 and abs(idx1 - idx2) <= self._max_frame_lookahead

