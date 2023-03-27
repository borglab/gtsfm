"""Loader for datasets used in 1DSFM and Colmap papers.

Authors: Yanwei Du
"""

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from gtsam import Cal3Bundler, Pose3

import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase

logger = logger_utils.get_logger()


class OneDSFMLoader(LoaderBase):
    """Loader for datasets used in 1DSFM and Colmap papers.

    The dataset is collected from internet and only partial images have exif data embedded. It can be downloaded from
    1DSFM [project website](https://www.cs.cornell.edu/projects/1dsfm/) and has the following folder structure:

        - images.SEQ_NAME/SEQ_NAME/images/*.jpg

    Reference:
    [1] Wilson, K., Snavely, N. (2014). Robust Global Translations with 1DSfM. ECCV 2014.
    https://doi.org/10.1007/978-3-319-10578-9_5.
    [2] J. L. Schönberger and J. -M. Frahm, "Structure-from-Motion Revisited," 2016 IEEE Conference on Computer Vision
    and Pattern Recognition (CVPR), 2016, pp. 4104-4113, doi: 10.1109/CVPR.2016.445.
    """

    def __init__(
        self,
        folder: str,
        image_extension: str = "jpg",
        max_resolution: int = 640,
        max_num_imgs: int = 0,
    ) -> None:
        """Initializes to load from a specified folder on disk.

        Args:
            folder: the base folder which contains image sequence.
            image_extension: file extension for the image files. Defaults to 'jpg'.
            max_resolution: integer representing maximum length of image's short side, i.e.
                the smaller of the height/width of the image. e.g. for 1080p (1920 x 1080),
                max_resolution would be 1080. If the image resolution max(height, width) is
                greater than the max_resolution, it will be downsampled to match the max_resolution.
            max_num_imgs: integer representing max number of images to process in the sequence,
                0 or negative values means to process all the images.
        """
        super().__init__(max_resolution=max_resolution)

        # Fetch all the file names in /images folder.
        search_path = os.path.join(folder, "images", f"*.{image_extension}")

        self._image_paths = glob.glob(search_path)
        num_imgs = len(self._image_paths)

        if max_num_imgs > 0 and max_num_imgs < num_imgs:
            random_indices = np.random.permutation(max_num_imgs)
            self._image_paths = np.array(self._image_paths)[random_indices[:max_num_imgs]]

            logger.info("Selected %d out of %d images.", max_num_imgs, num_imgs)

        self._num_imgs = len(self._image_paths)

        if self._num_imgs == 0:
            raise RuntimeError(f"Loader could not find any images with the specified file extension in {search_path}")

    def image_filenames(self) -> List[str]:
        """Return the file names corresponding to each image index."""
        return [Path(fpath).name for fpath in self._image_paths]

    def __len__(self) -> int:
        """The number of images in the dataset.

        Returns:
            The number of images.
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
            index: the index to fetch.

        Returns:
            Intrinsics for the given camera.
        """
        # Get intrinsics from exif.
        intrinsics = io_utils.load_image(self._image_paths[index]).get_intrinsics_from_exif()
        return intrinsics

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            The camera pose w_T_index.
        """
        # The 1DSfM datasets do not provide ground truth or prior camera poses.
        return None
