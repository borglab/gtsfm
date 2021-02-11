""" Loader that reads images from Argoverse directories on disk.

Authors: John Lambert
"""

import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
from gtsam import Cal3Bundler, Point3, Pose3, Rot3

from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.utils.calibration import get_calibration_config
from argoverse.utils.se3 import SE3

import gtsfm.utils.io as io_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase


class ArgoverseDatasetLoader(LoaderBase):
    """ """

    def __init__(self, dataset_dir: str, log_id: str, stride: int = 4) -> None:
        """
        Args:
            dataset_dir: directory where raw Argoverse logs are stored on disk
            log_id:
            stride: sampling rate, e.g. every 2 images, every 4 images, etc.
        """
        self.log_id = log_id
        self.dl = SimpleArgoverseTrackingDataLoader(
            data_dir=dataset_dir,
            labels_dir=dataset_dir
        )


        max_num_imgs = 20
        stride = 10 # images

        max_lookahead_sec = 60 # at original 30 fps frame rate, so 2 sec
        self.max_lookahead_imgs = max_lookahead_sec / stride

        self.calib_data = self.dl.get_log_calibration_data(log_id)
        camera_name = "ring_front_center"

        image_paths = self.dl.get_ordered_log_cam_fpaths(log_id, camera_name)
        image_paths = image_paths[::stride]
        self.image_paths = image_paths[:max_num_imgs]

        # for each image, cache its associated timestamp
        self.image_timestamps = [ int(Path(path).stem.replace(f'{camera_name}_', '')) for path in self.image_paths ]
 
        cam_config = get_calibration_config(self.calib_data, camera_name)
        self.K = cam_config.intrinsic[:3,:3]

        self.camera_SE3_egovehicle = SE3(
            rotation=cam_config.extrinsic[:3,:3],
            translation=cam_config.extrinsic[:3,3]
        )
        self.egovehicle_SE3_camera = self.camera_SE3_egovehicle.inverse()


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
        return Cal3Bundler(
            fx=min(self.K[0, 0], self.K[1, 1]),
            k1=0,
            k2=0,
            u0=self.K[0, 2],
            v0=self.K[1, 2],
        )

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        World coordinate frame is the "city" coordinate frame.

        Args:
            index: the index to fetch.

        Returns:
            the camera pose w_P_index.
        """
        timestamp = self.image_timestamps[index]
        city_SE3_egovehicle = self.dl.get_city_SE3_egovehicle(self.log_id, timestamp)
        if city_SE3_egovehicle is None:
            return None

        city_SE3_camera = city_SE3_egovehicle.compose(self.egovehicle_SE3_camera)

        return Pose3(
            Rot3(city_SE3_camera.rotation),
            Point3(city_SE3_camera.translation)
        )


    def validate_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """

        return (idx1 < idx2) and (idx2 < idx1 + self.max_lookahead_imgs)
