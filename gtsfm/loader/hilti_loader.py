"""Hilti dataset loader.

The dataset should be preprocessed to extract images from each camera into its respective folders.

Dataset ref: https://rpg.ifi.uzh.ch/docs/Arxiv21_HILTI.pdf
Kalibr format for intrinsics: https://github.com/ethz-asl/kalibr/wiki/yaml-formats

Authors: Ayush Baid
"""
import glob
from regex import P
import yaml
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from gtsam import Cal3Fisheye, Pose3

import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase

logger = logger_utils.get_logger()

AVAILABLE_CAM_IDXS = [0, 1, 2, 3, 4]

CAM_IDX_TO_KALIBR_FILE_MAP = {
    0: "calib_3_cam0-1-camchain-imucam.yaml",
    1: "calib_3_cam0-1-camchain-imucam.yaml",
    2: "calib_3_cam2-camchain-imucam.yaml",
    3: "calib_3_cam3-camchain-imucam.yaml",
    4: "calib_3_cam4-camchain-imucam.yaml",
}


class HiltiLoader(LoaderBase):
    def __init__(
        self,
        base_folder: str,
        cams_to_use: Set[int],
        max_resolution: int = 1000,
        max_frame_lookahead: int = 10,
        step_size: int = 10,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__(max_resolution=max_resolution)
        self.__check_cams_to_use(cams_to_use)
        self._base_folder: Path = Path(base_folder)
        self._cams_to_use: List[int] = list(cams_to_use)
        self._max_frame_lookahead: int = max_frame_lookahead
        self._step_size: int = step_size
        self._max_length = max_length
        self._calibrations: Dict[int, Cal3Fisheye] = {
            cam_idx: self.__load_calibration(cam_idx) for cam_idx in self._cams_to_use
        }
        self._number_of_timestamps_available: int = self.__get_number_of_timestamps_available()
        if self._max_length is not None:
            self._number_of_timestamps_available = min(self._number_of_timestamps_available, self._max_length)

        logger.info("Loading %d timestamps", self._number_of_timestamps_available)

    def __get_folder_for_images(self, cam_idx: int) -> Path:
        return self._base_folder / f"cam{cam_idx}"

    def __check_cams_to_use(self, cams_to_use: Set[int]) -> None:
        for cam_idx in cams_to_use:
            assert cam_idx in AVAILABLE_CAM_IDXS

    def __get_number_of_timestamps_available(self) -> int:
        num_images_for_cam = {}
        for cam_idx in self._cams_to_use:
            search_path: str = str(self._base_folder / f"cam{cam_idx}" / "*.jpg")
            image_files = glob.glob(search_path)
            num_images_for_cam[cam_idx] = len(image_files)

        max_num_images = max(num_images_for_cam.values())

        return max_num_images // self._step_size

    def __load_calibration(self, cam_idx: int) -> Cal3Fisheye:
        kalibr_file_path = self._base_folder / "calibration" / CAM_IDX_TO_KALIBR_FILE_MAP[cam_idx]

        with open(kalibr_file_path, "r") as file:
            calibration_data = yaml.safe_load(file)
            if cam_idx != 1:
                calibration_data = calibration_data["cam0"]
            else:
                calibration_data = calibration_data["cam1"]

            assert calibration_data["camera_model"] == "pinhole"
            assert calibration_data["distortion_model"] == "equidistant"

            intrinsics: Cal3Fisheye = self.__load_intrinsics(calibration_data)

        return intrinsics

    def __load_intrinsics(self, calibration_data: Dict[Any, Any]) -> Cal3Fisheye:
        fx, fy, px, py = calibration_data["intrinsics"]
        k1, k2, k3, k4 = calibration_data["distortion_coeffs"]

        return Cal3Fisheye(fx=fx, fy=fy, s=0, u0=px, v0=py, k1=k1, k2=k2, k3=k3, k4=k4)

    def __len__(self) -> int:
        """
        The number of images in the dataset.

        Returns:
            the number of images.
        """
        return self._number_of_timestamps_available * len(self._cams_to_use)

    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            Image: the image at the query index.
        """
        cam_for_index = self.__map_index_to_camera(index)
        # TODO: move this logic to a function. Also, select better names
        image_index_for_cam = index // len(self._cams_to_use) * self._step_size

        logger.debug("Mapping %d index to image %d of camera %d", index, image_index_for_cam, cam_for_index)

        camera_folder: Path = self.__get_folder_for_images(cam_for_index)
        image_path: Path = camera_folder / f"left{image_index_for_cam:04}.jpg"

        return io_utils.load_image(str(image_path))

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3Fisheye]:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        return self._calibrations[self.__map_index_to_camera(index)]

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            the camera pose w_P_index.
        """
        return None

    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair. idx1 < idx2 is required.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """
        return super().is_valid_pair(idx1, idx2) and abs(idx1 - idx2) <= self._max_frame_lookahead

    def __map_index_to_camera(self, index: int) -> int:
        return self._cams_to_use[index % len(self._cams_to_use)]


if __name__ == "__main__":
    root = "/media/ayush/cross_os1/dataset/hilti"

    loader = HiltiLoader(root, {1, 3, 4})

    for i in range(100):
        loader.get_image(i)
