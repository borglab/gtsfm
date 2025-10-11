"""Loader that reads images from Argoverse directories on disk.

Authors: John Lambert
"""

from pathlib import Path
from typing import Optional

import numpy as np
from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.utils.calibration import get_calibration_config
from argoverse.utils.se3 import SE3
from gtsam import Cal3Bundler, Pose3, Rot3

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
        stride: int = 5,
        max_num_imgs: int = 20,
        max_lookahead_sec: float = 2,
        camera_name: str = "ring_front_center",
        max_resolution: int = 760,
    ) -> None:
        """Select the image paths and their corresponding timestamps for images to feed to GTSFM.
        Args:
            dataset_dir: directory where raw Argoverse logs are stored on disk
            log_id: unique ID of vehicle log
            stride: sampling rate, e.g. every 2 images, every 4 images, etc.
            max_num_imgs: number of images to load (starting from beginning of log sequence)
            max_resolution: integer representing maximum length of image's short side, i.e.
               the smaller of the height/width of the image. e.g. for 1080p (1920 x 1080),
               max_resolution would be 1080. If the image resolution max(height, width) is
               greater than the max_resolution, it will be downsampled to match the max_resolution.
        """
        super().__init__(max_resolution)
        self._log_id = log_id
        self._dl = SimpleArgoverseTrackingDataLoader(data_dir=dataset_dir, labels_dir=dataset_dir)
        self.load_camera_calibration(log_id, camera_name)

        # computed as #frames = #sec * (frame/sec)
        max_lookahead_num_fr = max_lookahead_sec * RING_CAMERA_FRAME_RATE

        # in subsampled list, account for subsampling rate
        self._max_lookahead_for_img = max_lookahead_num_fr / stride

        image_paths = self._dl.get_ordered_log_cam_fpaths(log_id, camera_name)
        image_timestamps = [int(Path(path).stem.split("_")[-1]) for path in image_paths]
        # only choose frames where ground truth egovehicle pose is provided
        valid_idxs = [
            idx for idx, ts in enumerate(image_timestamps) if self._dl.get_city_SE3_egovehicle(log_id, ts) is not None
        ]
        image_paths = [image_paths[idx] for idx in valid_idxs]
        image_timestamps = [image_timestamps[idx] for idx in valid_idxs]

        # subsample, then limit the total size of dataset
        image_paths = image_paths[::stride]
        self._image_paths = image_paths[:max_num_imgs]

        # for each image, cache its associated timestamp
        image_timestamps = image_timestamps[::stride]
        self._image_timestamps = image_timestamps[:max_num_imgs]

        # set the first pose as origin
        self._world_pose = Pose3(Rot3(), np.zeros((3, 1)))
        self._world_pose = self.get_camera_pose(0)

    def image_filenames(self) -> list[str]:
        """Return the file names corresponding to each image index."""
        return [Path(path).name for path in self._image_paths]

    def load_camera_calibration(self, log_id: str, camera_name: str) -> None:
        """Load extrinsics and intrinsics from disk."""
        calib_data = self._dl.get_log_calibration_data(log_id)
        cam_config = get_calibration_config(calib_data, camera_name)
        self._K = cam_config.intrinsic[:3, :3]

        # square pixels, so fx and fy should be (nearly) identical
        assert np.isclose(self._K[0, 0], self._K[1, 1], atol=0.1)

        self._camera_SE3_egovehicle = SE3(
            rotation=cam_config.extrinsic[:3, :3], translation=cam_config.extrinsic[:3, 3]
        )
        self._egovehicle_SE3_camera = self._camera_SE3_egovehicle.inverse()

    def __len__(self) -> int:
        """The number of images in the dataset.

        Returns:
            The number of images.
        """
        return len(self._image_paths)

    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            The image at the query index.
        """

        if index < 0 or index >= len(self):
            raise IndexError("Image index is invalid")

        return io_utils.load_image(self._image_paths[index])

    def get_camera_intrinsics_full_res(self, index: int) -> Cal3Bundler:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            the index to fetch.

        Returns:
            Intrinsics for the given camera.
        """
        return Cal3Bundler(
            fx=self._K[0, 0],
            k1=0,
            k2=0,
            u0=self._K[0, 2],
            v0=self._K[1, 2],
        )

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Note: the world coordinate frame is the "city" coordinate frame.

        Args:
            index: the index to fetch.

        Returns:
            The camera pose wTi, where `i` represents image/frame `index`
        """
        timestamp = self._image_timestamps[index]
        city_SE3_egovehicle = self._dl.get_city_SE3_egovehicle(self._log_id, timestamp)
        assert city_SE3_egovehicle is not None
        city_SE3_camera = city_SE3_egovehicle.compose(self._egovehicle_SE3_camera)

        return self._world_pose.between(Pose3(Rot3(city_SE3_camera.rotation), city_SE3_camera.translation))

    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair. idx1 < idx2 is required.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """
        return super().is_valid_pair(idx1, idx2) and (idx2 < idx1 + self._max_lookahead_for_img)
