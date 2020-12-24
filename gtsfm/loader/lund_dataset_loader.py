"""Data loader for the Lund dataset.

Authors: Ayush Baid
"""

import os
from typing import List

import numpy as np
import scipy.io as scipy_io

from gtsfm.loader.folder_loader import FolderLoader


class LundDatasetLoader(FolderLoader):
    """Simple loader class that reads from a folder on disk.

    Folder layout structure:
    - RGB Images: images/
    - Extrinsics data (optional): extrinsics/
        - numpy array with the same name as images


    If explicit intrinsics are not provided, the exif data will be used.
    """

    def __init__(self, folder: str) -> None:
        """Initialize object to load image data from a specified folder on disk

        Args:
            folder: the base folder for a given scene.
        """

        self.folder_base = folder

        super().__init__(folder, image_extension='JPG')

        # construct the extrinsics if they do not exist already
        if not self.explicit_extrinsics_paths:
            self.explicit_extrinsics_paths = \
                self.__generate_extrinsics_from_reconstruction()

    def __generate_extrinsics_from_reconstruction(self) -> List[str]:
        """Extract extrinsics from mat file and stores them as numpy arrays.

        The reconstruction used for extrinsics is provided by Carl Olsson as 
        part of the Lund dataset.

        Returns:
            file names of generated extrinsics for each pose.
        """

        reconstruction_path = os.path.join(
            self.folder_base, 'reconstruction', 'data.mat')

        extrinsics_path_template = os.path.join(
            self.folder_base, 'extrinsics', '{}.npy'
        )

        loaded_data = scipy_io.loadmat(reconstruction_path)

        image_names = loaded_data['imnames']
        poses = loaded_data['P'][0]

        num_images = image_names.shape[0]

        filenames = []

        for idx in range(num_images):
            # 2nd indexing is a dummy index (as everything stored as arrays)
            # 3rd indexing is for array with name as opposed to other metadata
            # 4th indexing finally accesses the name from the singleton array.
            image_name = image_names[idx][0][0][0]

            image_name = os.path.splitext(image_name)[0]  # remove the extension
            extrinsics = poses[idx]
            filename = extrinsics_path_template.format(image_name)
            np.save(filename, extrinsics)

            filenames.append(filename)

        return filenames
