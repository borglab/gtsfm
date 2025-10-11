"""Loader for the Image-Matching-Benchmark's YFCC dataset.

References: https://www.cs.ubc.ca/research/image-matching-challenge/

Authors: Ayush Baid
"""
import os.path as osp
from typing import List, Optional

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3

import gtsfm.utils.io as io_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase


class YfccImbLoader(LoaderBase):
    """Loader for IMB's YFCC dataset.

    Code ref: https://github.com/vcg-uvic/image-matching-benchmark/blob/master/compute_stereo.py
    """

    def __init__(self, dataset_dir: str, co_visibility_threshold: float = 0.1, max_resolution: int = 760) -> None:
        """Initializes the loader.

        Args:
            dataset_dir: the base dataset directory.
            co_visibility_threshold (optional): threshold for co-visibility between two images to be considered valid.
                                     Defaults to 0.1.
            max_resolution: integer representing maximum length of image's short side, i.e.
               the smaller of the height/width of the image. e.g. for 1080p (1920 x 1080),
               max_resolution would be 1080. If the image resolution max(height, width) is
               greater than the max_resolution, it will be downsampled to match the max_resolution.
        """
        super().__init__(max_resolution)
        self._dataset_dir = dataset_dir

        # load all the image pairs according to the co-visibility threshold used
        # in IMB's reporting
        visibility_file = osp.join(
            self._dataset_dir,
            "new-vis-pairs",
            "keys-th-{:0.1f}.npy".format(coviz_thresh),
        )

        image_pairs = list()  # list of image pairs
        image_names = set()  # set of image names (for uniqueness)
        for pair_entry in np.load(visibility_file):
            file1, file2 = pair_entry.split("-")

            image_pairs.append((file1, file2))

            image_names.add(file1)
            image_names.add(file2)

        # convert image names from set to list
        self._image_names: List[str] = sorted(list(image_names))

        # map image names to its position in the list
        self._name_to_idx_map = {name: i for i, name in enumerate(self._image_names)}

        self._image_pairs = set()
        for i1, i2 in image_pairs:
            if i1 < i2:
                self._image_pairs.add((i1, i2))
            else:
                self._image_pairs.add((i2, i1))

        self._cameras = self.__read_calibrations()  # self.__read_colmap_model()

    def __len__(self) -> int:
        """
        The number of images in the dataset.

        Returns:
            the number of images.
        """
        return len(self._image_names)

    def image_filenames(self) -> List[str]:
        """Return the file names corresponding to each image index."""
        return self._image_names

    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            Image: the image at the query index.
        """
        if index < 0 or index > self.__len__():
            raise IndexError("Image index is invalid")

        image_name = self._image_names[index]

        file_name = osp.join(self._dataset_dir, "images", "{}.jpg".format(image_name))

        return io_utils.load_image(file_name)

    def get_camera_intrinsics_full_res(self, index: int) -> Cal3Bundler:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Image index is invalid")

        return self._cameras[index].calibration()

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            the camera pose w_P_index.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Image index is invalid")

        return self._cameras[index].pose()

    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair. idx1 < idx2 is required.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """
        return super().is_valid_pair(idx1, idx2) and (idx1, idx2) in self._image_pairs

    def __read_calibrations(self) -> List[PinholeCameraCal3Bundler]:
        """Read camera params from the calibration stored as h5 files.

        Returns:
            list of all cameras.
        """

        file_path_template = osp.join(self._dataset_dir, "calibration", "calibration_{}.h5")

        pose_list = []

        for image_name in self._image_names:
            file_path = file_path_template.format(image_name)
            calib_data = io_utils.load_h5(file_path)

            cTw = Pose3(Rot3(calib_data["R"]), calib_data["T"])
            K_matrix = calib_data["K"]
            # TODO: fix different fx and fy (and underparameterization of K)
            K = Cal3Bundler(
                fx=float(K_matrix[0, 0] + K_matrix[1, 1]) * 0.5,
                k1=0.0,
                k2=0.0,
                u0=float(K_matrix[0, 2]),
                v0=float(K_matrix[1, 2]),
            )

            pose_list.append(PinholeCameraCal3Bundler(cTw.inverse(), K))

        return pose_list
