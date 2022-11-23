"""Loader of dataset used in 1DSFM and Colmap paper

The dataset is collected from internet and has EXIF embedded in each image,
it contains several sequences with naming format: images.xxx/

Folder sturcutre of a single sequence:
- images/*.jpg

Ref: https://www.cs.cornell.edu/projects/1dsfm/
Paper 1: https://www.cs.cornell.edu/projects/1dsfm/docs/1DSfM_ECCV14.pdf
Paper 2: https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf

Authors: Yanwei Du

"""

import glob
import numpy as np
import os
from pathlib import Path
from typing import List, Optional

import scipy.io
from gtsam import Cal3Bundler, Pose3

import gtsfm.utils.io as io_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase


class OneDSFMLoader(LoaderBase):
    def __init__(
        self,
        folder: str,
        image_extension: str = "jpg",
        max_resolution: int = 640,
        max_num_imgs: int = 0
    ) -> None:
        """Initializes to load from a specified folder on disk.

        Args:
            folder: the base folder which contains image sequence
            image_extension: file extension for the image files. Defaults to 'jpg'.
            max_resolution: integer representing maximum length of image's short side, i.e.
                the smaller of the height/width of the image. e.g. for 1080p (1920 x 1080),
                max_resolution would be 1080. If the image resolution max(height, width) is
                greater than the max_resolution, it will be downsampled to match the max_resolution.
            max_num_imgs: integer representing max number of images to process in the sequence,
                0 or negative values means to process all the images
        """
        super().__init__(max_resolution=max_resolution)

        # fetch all the file names in /images folder
        search_path = os.path.join(folder, 'images', f"*.{image_extension}")

        # @NOTE for now we only process images with valid EXIF data
        self._image_paths, self._num_exif_imgs, self._num_all_imgs = \
            self.get_images_with_valid_exif(search_path, max_num_imgs)
        self._num_imgs = len(self._image_paths)

        # self._image_paths.sort()
        print(f'selected {self._num_imgs} out of {self._num_exif_imgs} images with valid exif')

        if self._num_imgs == 0:
            raise RuntimeError(
                f"Loader could not find any images with the specified file extension in {search_path}")

    def get_images_with_valid_exif(self, search_path: str, max_num_imgs: int) -> List[str]:
        """Return a subset of images with valid exif.

        Args:
            search_path: image sequence search path
            max_num_imgs: the maximum number of images to process

        Returns:
            list of selected image path.
            the number of images with valid exif data.
            the number of all the images.
        """

        all_image_paths = glob.glob(search_path)
        num_all_imgs = len(all_image_paths)
        exif_image_paths = []
        for p in all_image_paths:
            # drop images without valid exif data
            if io_utils.load_image(p).get_intrinsics_from_exif():
                exif_image_paths.append(p)

        # randomly select a subset to process
        num_exif_imgs = len(exif_image_paths)
        max_num_imgs = max_num_imgs if (max_num_imgs > 0 and max_num_imgs < num_exif_imgs) else num_exif_imgs
        random_indices = np.random.permutation(num_exif_imgs)
        return np.array(exif_image_paths)[random_indices[:max_num_imgs]], num_exif_imgs, num_all_imgs

    def image_filenames(self) -> List[str]:
        """Return the file names corresponding to each image index."""
        return [Path(fpath).name for fpath in self._image_paths]

    def __len__(self) -> int:
        """The number of images in the dataset.

        Returns:
            the number of images.
        """
        return self._num_imgs

    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            Image: the image at the query index.
        """

        if index < 0 or index >= len(self):
            raise IndexError("Image index is invalid")

        return io_utils.load_image(self._image_paths[index])

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3Bundler]:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        # get intrinsics from exif
        intrinsics = io_utils.load_image(
            self._image_paths[index]).get_intrinsics_from_exif()

        return intrinsics

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            the camera pose w_T_index.
        """

        # No prior pose is available for images collected from Internet
        return None

    def get_num_all_imgs(self) -> int:
        """Return the number of all images in the sequence"""

        return self._num_all_imgs

    def get_num_exif_imgs(self) -> int:
        """Return the number of images with valid exif data"""

        return self._num_exif_imgs