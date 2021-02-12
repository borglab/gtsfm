"""Loader that reads images from Argoverse directories on disk.

Authors: John Lambert
"""

from pathlib import Path
from typing import Optional

import numpy as np
from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.utils.calibration import get_calibration_config
from argoverse.utils.se3 import SE3
from gtsam import Cal3Bundler, Point3, Pose3, Rot3

import gtsfm.utils.io as io_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase

RING_CAMERA_FRAME_RATE = 30  # fps or Hz


class ArgoverseDatasetLoader(LoaderBase):
    """ """

    def __init__(
        self,
        dataset_dir: str,
        log_id: str,
        stride: int = 10,
        max_num_imgs: int = 20,
        max_lookahead_sec: float = 2,
        camera_name: str = "ring_front_center",
    ) -> None:
        """
        Args:
            dataset_dir: directory where raw Argoverse logs are stored on disk
            log_id: unique ID of vehicle log
            stride: sampling rate, e.g. every 2 images, every 4 images, etc.
            max_num_imgs: number of images to load (starting from beginning of log sequence)
        """
        self.log_id_ = log_id
        self.dl_ = SimpleArgoverseTrackingDataLoader(data_dir=dataset_dir, labels_dir=dataset_dir)

        # computed as #frames = #sec * (frame/sec)
        max_lookahead_num_fr = max_lookahead_sec * RING_CAMERA_FRAME_RATE

        # in subsampled list, account for subsampling rate
        self.max_lookahead_for_img_ = max_lookahead_num_fr / stride

        self.calib_data_ = self.dl_.get_log_calibration_data(log_id)

        image_paths = self.dl_.get_ordered_log_cam_fpaths(log_id, camera_name)
        image_paths = image_paths[::stride]
        self.image_paths_ = image_paths[:max_num_imgs]

        # for each image, cache its associated timestamp
        self.image_timestamps_ = [int(Path(path).stem.replace(f"{camera_name}_", "")) for path in self.image_paths_]

        cam_config = get_calibration_config(self.calib_data_, camera_name)
        self.K_ = cam_config.intrinsic[:3, :3]

        self.camera_SE3_egovehicle_ = SE3(
            rotation=cam_config.extrinsic[:3, :3], translation=cam_config.extrinsic[:3, 3]
        )
        self.egovehicle_SE3_camera_ = self.camera_SE3_egovehicle_.inverse()

    def __len__(self) -> int:
        """The number of images in the dataset.

        Returns:
            The number of images.
        """
        return len(self.image_paths_)

    def get_image(self, index: int) -> Image:
        """Get the image at the given index.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            The image at the query index.
        """

        if index < 0 or index >= self.__len__():
            raise IndexError("Image index is invalid")

        return io_utils.load_image(self.image_paths_[index])

    def get_camera_intrinsics(self, index: int) -> Optional[Cal3Bundler]:
        """Get the camera intrinsics at the given index.

        Args:
            the index to fetch.

        Returns:
            Intrinsics for the given camera.
        """
        return Cal3Bundler(
            fx=min(self.K_[0, 0], self.K_[1, 1]),
            k1=0,
            k2=0,
            u0=self.K_[0, 2],
            v0=self.K_[1, 2],
        )

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Note: the world coordinate frame is the "city" coordinate frame.

        Args:
            index: the index to fetch.

        Returns:
            The camera pose wTi, where `i` represents image/frame `index`
        """
        timestamp = self.image_timestamps_[index]
        city_SE3_egovehicle = self.dl_.get_city_SE3_egovehicle(self.log_id_, timestamp)
        if city_SE3_egovehicle is None:
            return None

        city_SE3_camera = city_SE3_egovehicle.compose(self.egovehicle_SE3_camera_)

        return Pose3(Rot3(city_SE3_camera.rotation), city_SE3_camera.translation)

    def validate_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """

        return (idx1 < idx2) and (idx2 < idx1 + self.max_lookahead_for_img_)
