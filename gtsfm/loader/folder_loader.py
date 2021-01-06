""" Simple loader that reads from a folder on disk.

Authors: Frank Dellaert and Ayush Baid
"""

import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
from gtsam import Cal3Bundler, Pose3

import gtsfm.utils.io as io_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase


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

    def __init__(self, folder: str, image_extension: str = 'jpg') -> None:
        """
        Initializes to load from a specified folder on disk

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

        self.explicit_intrinsics_paths = []
        for image_file_name in self.image_paths:
            file_path = os.path.join(
                folder, 'intrinsics', '{}.npy'.format(
                    os.path.splitext(os.path.basename(image_file_name))[0]
                ))
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
                Path(image_file_name).stem
            )
            if not os.path.exists(file_path):
                self.explicit_extrinsics_paths = []
                break
            else:
                self.explicit_extrinsics_paths.append(file_path)

    def __len__(self) -> int:
        """
        The number of images in the dataset.

        Returns:
            the number of images.
        """
        return len(self.image_paths)

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

    def get_camera_intrinsics(self, index: int) -> Optional[Cal3Bundler]:
        """Get the camera intrinsics at the given index.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        if len(self.explicit_intrinsics_paths) == 0:
            # get intrinsics from exif

            return io_utils.load_image(self.image_paths[index]).get_intrinsics_from_exif()

        else:
            # TODO: handle extra inputs in the intrinsics array
            intrinsics_array = np.load(self.explicit_intrinsics_paths[index])

            return Cal3Bundler(
                fx=min(intrinsics_array[0, 0], intrinsics_array[1, 1]),
                k1=0,
                k2=0,
                u0=intrinsics_array[0, 2],
                v0=intrinsics_array[1, 2])

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            the camera pose w_P_index.
        """
        if self.explicit_extrinsics_paths:
            numpy_extrinsics = np.load(self.explicit_extrinsics_paths[index])

            return Pose3(numpy_extrinsics)

        return None

    def validate_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """

        return idx1 < idx2 and idx2 < idx1 + 4
