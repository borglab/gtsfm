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

        self._K, self._camera_poses = self.__read_camera_params_from_reconstruction()

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

        # rotation between the 0^th camera and world system
        cam_rotation = Rot3.RzRyRx(-np.pi / 2, 0, 0)
        cam_coordinate_pose = Pose3(cam_rotation, np.zeros((3,)))

        camera_poses = [Pose3()]  # adding the first pose

        for idx in range(1, num_images):
            P = projection_matrices[idx]
            extrinsics = np.linalg.inv(K_matrix) @ P  # this looks fishy. extrinsics constraints not withheld.

            camera_poses.append(Pose3(extrinsics))

        # transform the poses so that cameras point in the right direction.
        camera_poses = [p.between(cam_coordinate_pose) for p in camera_poses]

        return K, camera_poses

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
        return self._camera_poses[index]


if __name__ == "__main__":
    path = "data/lund/door"

    loader = LundDatasetLoader(path)

    for i in range(len(loader)):
        intrinsics_mat = loader.get_camera_intrinsics(i).K()
        np.save("data/lund/door/intrinsics/{}.npy".format(i), intrinsics_mat)

        extrinsics_mat = loader.get_camera_pose(i).matrix()
        np.save("data/lund/door/extrinsics/{}.npy".format(i), extrinsics_mat)

    # import gtsfm.utils.viz as viz_utils
    # import matplotlib.pyplot as plt

    # cam_poses = [loader.get_camera_pose(i) for i in range(len(loader))]

    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # viz_utils.plot_poses_3d(cam_poses, ax)
    # plt.show()
