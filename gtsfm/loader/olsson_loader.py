"""Simple loader class that reads Carl Olsson's dataset from a folder on disk.

Authors: John Lambert
"""

import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
from gtsam import Cal3Bundler, Pose3
from scipy.io import loadmat

import gtsfm.utils.io as io_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase


class OlssonLoader(LoaderBase):
    """Simple loader class that reads Carl Olsson's dataset from a folder on disk.

    Ref: http://www.maths.lth.se/matematiklth/personal/calle/dataset/dataset.html

    Folder layout structure:
    - RGB Images: images/
    - Intrinsics + Extrinsics data (optional): data.mat

    If explicit intrinsics are not provided, the exif data will be used.
    """

    def __init__(self, folder: str, image_extension: str = "jpg", use_gt_intrinsics: bool = True) -> None:
        """
        Initializes to load from a specified folder on disk

        Args:
            folder: the base folder for a given scene
            image_extension: (optional)extension for the image files. Defaults to 'jpg'.
            use_gt_intrinsics: whether to use ground truth intrinsics
        """
        self.use_gt_intrinsics = use_gt_intrinsics

        # fetch all the file names in /images folder
        search_path = os.path.join(folder, "images", f"*.{image_extension}")

        self.image_paths = glob.glob(search_path)

        # sort the file names
        self.image_paths.sort()
        num_imgs = len(self.image_paths)

        file_path = os.path.join(folder, "data.mat")
        if not Path(file_path).exists():
            # not available, so no choice
            self.use_gt_intrinsics = False
            return

        # stores camera poses (extrinsics) and intrinsics as 3x4 projection matrices
        # Array will have shape (1,num_imgs), and each element will be a (3,4) matrix
        data = loadmat(file_path)

        # M = K [R | t]
        # in GTSAM notation, M = K @ cTw
        M_list = [data['P'][0][i] for i in range(num_imgs)]

        # first pose is identity, so K is immediate given
        self.K = M_list[0][:3,:3]
        Kinv = np.linalg.inv(self.K)

        # decode camera poses as:
        # # K^{-1} @ M = cTw
        self.iTw_list = [ Kinv @ M_list[i] for i in range(num_imgs)]


    def __len__(self) -> int:
        """
        The number of images in the dataset.

        Returns:
            the number of images.
        """
        return self.num_imgs

    def get_image(self, index: int) -> Image:
        """
        Get the image at the given index.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            Image: the image at the query index.
        """

        if index < 0 or index > self.__len__():
            raise IndexError("Image index is invalid")

        return io_utils.load_image(self.image_paths[index])


    def get_camera_intrinsics(self, index: int) -> Cal3Bundler:
        """Get the camera intrinsics at the given index.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        if not self.use_gt_intrinsics:
            # get intrinsics from exif
            intrinsics = io_utils.load_image(self.image_paths[index]).get_intrinsics_from_exif()

        else:
            intrinsics = Cal3Bundler(
                fx=min(self.K[0, 0], self.K[1, 1]),
                k1=0,
                k2=0,
                u0=self.K[0, 2],
                v0=self.K[1, 2],
            )
        return intrinsics


    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            the camera pose w_T_index.
        """
        if not self.use_gt_intrinsics:
            # poses also not available
            return None

        iTw = self.iTw_list[index]
        iTw = Pose3(iTw[:3,:3], iTw[:,3])

        return iTw.inverse()


    def validate_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """

        return idx1 < idx2

