""" Simple loader that reads from a folder on disk.

Authors: Frank Dellaert and Ayush Baid
"""

import glob
import os

import numpy as np

import utils.io as io_utils
from common.image import Image
from loader.loader_base import LoaderBase


class FolderLoader(LoaderBase):
    """Simple loader class that reads from a folder on disk.

    Folder layout structure:
    - RGB Images: images/
    - Intrinsics data (optional): intrinsics/ 
        - numpy arrays with the same name as images
    - Extrinsics data (optional): extrinsics/
        - numpy array with the same name as images


    If explicit intrinsics are not provided, the exif data will be used.
    """

    def __init__(self, folder: str, image_extension: str = 'jpg'):
        """
        Initializes to load from a specified folder on disk

        Folder structure:
        - /images: the image files in the specified extension

        Args:
            folder (str): the base folder for a given scene
            image_extension (str, optional): extension for the image files. Defaults to 'jpg'.
        """

        # fetch all the file names in /image folder
        search_path = os.path.join(
            folder, 'images', '*.{}'.format(image_extension)
        )

        self.image_paths = glob.glob(search_path)

        # sort the file names
        self.image_paths.sort()

        # check if intrisincs are available as numpy arrays
        explicit_intrinsics_template = os.path.join(
            folder, 'intrinsics', '{}.npy'
        )

        self.explicit_intrinsics_paths = []
        for image_file_name in self.image_paths:
            file_path = explicit_intrinsics_template.format(
                os.path.splitext(os.path.basename(image_file_name))[0]
            )
            if not os.path.exists(file_path):
                self.explicit_intrinsics_paths = []
                break
            else:
                self.explicit_intrinsics_paths.append(file_path)

        # check if extrinsics are available as numpy arrays
        explicit_extrinsics_template = os.path.join(
            folder, 'extrinsics', '{}.npy'
        )

        self.explicit_extrinsics_paths = []
        for image_file_name in self.image_paths:
            file_path = explicit_extrinsics_template.format(
                os.path.basename(image_file_name).split('.')[0]
            )
            if not os.path.exists(file_path):
                self.explicit_extrinsics_template = []
                break
            else:
                self.explicit_extrinsics_paths.append(file_path)

    def __len__(self) -> int:
        """
        Returns the number of images in the folder

        Returns:
            int: the number of images in the folder
        """
        return len(self.image_paths)

    def get_image(self, index: int) -> Image:
        """
        Get the image at the given index

        Args:
            index (int): the index to fetch

        Raises:
            IndexError: if an out-of-bounds image index is requested

        Returns:
            Image: the image at the query index
        """

        if index < 0 or index > self.__len__():
            raise IndexError("Image index is invalid")

        return io_utils.load_image(self.image_paths[index])

    def get_geometry(self, idx1: int, idx2: int) -> np.ndarray:
        """Get the ground truth fundamental matrix/homography from idx1 to idx2.

        The function returns either idx1_F_idx2 or idx1_H_idx2

        Args:
            idx1 (int): one of the index
            idx2 (int): one of the index

        Returns:
            np.ndarray: fundamental matrix/homograph matrix
        """

        return None

    def get_camera_intrinsics(self, index: int) -> np.ndarray:
        """Get the camera intrinsics at the given index.

        Args:
            index (int): the index to fetch

        Returns:
            np.ndarray: the 3x3 intrinsics matrix of the camera
        """
        if len(self.explicit_intrinsics_paths) == 0:
            # get intrinsics from exif

            return io_utils.load_image(self.image_paths[index]).get_intrinsics_from_exif()

        else:
            return np.load(self.explicit_intrinsics_paths[index])

    def get_camera_extrinsics(self, index: int) -> np.ndarray:
        """Get the camera extrinsics (pose) at the given index.

        The extrinsics format is [wRc, wTc]

        Args:
            index (int): the index to fetch

        Returns:
            np.ndarray: the 3x4 extrinsics matrix of the camer
        """
        if self.explicit_extrinsics_paths:
            return np.load(self.explicit_extrinsics_paths[index])
