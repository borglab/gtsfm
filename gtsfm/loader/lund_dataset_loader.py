"""Data loader for the Lund dataset.

Authors: Ayush Baid
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import scipy.io as scipy_io
from gtsam import Cal3Bundler, Pose3, Rot3

from gtsfm.loader.folder_loader import FolderLoader


class LundDatasetLoader(FolderLoader):
    """Simple loader class that reads from a folder on disk.

    Folder layout structure:
    - RGB Images: images/
    - Extrinsics data (optional): extrinsics/
        - numpy array with the same name as images


    If explicit intrinsics are not provided, the exif data will be used.
    """

    # TODO: fix the issues in other scenes with portrait and landscape.

    def __init__(self, folder: str) -> None:
        """Initialize object to load image data from a specified folder on disk

        Args:
            folder: the base folder for a given scene.
        """

        self.folder_base = folder

        super().__init__(folder, image_extension="JPG")

        self._K, self._wTc = self.__read_camera_params_from_reconstruction()

    def __read_camera_params_from_reconstruction(self) -> Tuple[Cal3Bundler, List[Pose3]]:
        """Extract extrinsics from mat file and stores them as numpy arrays.

        The reconstruction used for extrinsics is provided by Carl Olsson as part of the Lund dataset.

        Returns:
            file names of generated extrinsics for each pose.
        """

        reconstruction_path = os.path.join(self.folder_base, "reconstruction", "data.mat")

        loaded_data = scipy_io.loadmat(reconstruction_path)

        projection_matrices = loaded_data["P"][0]
        num_images = len(projection_matrices)

        # read intrinsics from first image
        P = projection_matrices[0]
        K_matrix = P[:, :3]
        K = Cal3Bundler(
            fx=float(K_matrix[0, 0] + K_matrix[1, 1]) * 0.5,
            k1=0.0,
            k2=0.0,
            u0=float(K_matrix[0, 2]),
            v0=float(K_matrix[1, 2]),
        )

        poses_wTc = [Pose3()]  # adding the first pose

        for idx in range(1, num_images):
            P = projection_matrices[idx]
            extrinsics_cTw = np.linalg.inv(K_matrix) @ P  # this looks fishy. extrinsics constraints not withheld.

            poses_wTc.append(Pose3(extrinsics_cTw).inverse())

        return K, poses_wTc

    def get_camera_intrinsics(self, index: int) -> Optional[Cal3Bundler]:
        """Get the camera intrinsics at the given index.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        return self._K

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            the camera pose w_P_index.
        """
        return self._wTc[index]

