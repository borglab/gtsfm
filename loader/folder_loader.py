""" Simple loader that reads from a folder on disk.

Authors: Frank Dellaert and Ayush Baid
"""

import glob
import os

import utils.io as io_utils
from common.image import Image
from loader.loader_base import LoaderBase


class FolderLoader(LoaderBase):
    """Simple loader class that reads from a folder on disk."""

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

        self.file_names = glob.glob(search_path)

        # sort the file names
        self.file_names.sort()

    def __len__(self) -> int:
        """
        Returns the number of images in the folder

        Returns:
            int: the number of images in the folder
        """
        return len(self.file_names)

    def get_image(self, index) -> Image:
        """
        Get the image at the given index

        Args:
            index (int): the index to fetch

        Returns:
            Image: the image at the query index
        """
        if index < 0 or index > self.__len__():
            raise IndexError("Image index is invalid")

        return io_utils.load_image(self.file_names[index])
