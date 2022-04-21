"""Hilti dataset loader.

The dataset should be preprocessed to extract images from each camera into its respective folders.

Dataset ref: https://rpg.ifi.uzh.ch/docs/Arxiv21_HILTI.pdf
Kalibr format for intrinsics: https://github.com/ethz-asl/kalibr/wiki/yaml-formats

Authors: Ayush Baid
"""
import os
import glob
import yaml
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

import numpy as np
from gtsam import Cal3Fisheye, Pose3, Rot3
from scipy.spatial.transform import Rotation as scipyR


import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.common.pose_prior import PosePrior, PosePriorType
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
        self._fastlio_output_path: Path = Path(self._base_folder, "fastlio2_odom.txt")
        self._cams_to_use: List[int] = list(cams_to_use)
        self._max_frame_lookahead: int = max_frame_lookahead
        self._step_size: int = step_size
        self._max_length = max_length
        self._intrinsics: Dict[int, Cal3Fisheye] = {}
        self._cam_T_imu_poses: Dict[int, Pose3] = {}
        for cam_idx in self._cams_to_use:
            calibration = self.__load_calibration(cam_idx)
            self._intrinsics[cam_idx] = calibration[0]
            self._cam_T_imu_poses[cam_idx] = calibration[1]

        self._number_of_timestamps_available: int = self.__get_number_of_timestamps_available()
        # self._timestamp_to_absolute_pose = self.
        if self._max_length is not None:
            self._number_of_timestamps_available = min(self._number_of_timestamps_available, self._max_length)

        self.timestamp_to_pose = self.store_fastlio_output()


        # import matplotlib.pyplot as plt
        # import gtsfm.utils.viz as viz_utils

        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")

        # poses = [self._cam_T_imu_poses[i] for i in self._cams_to_use]
        # pose_0 = poses[0].inverse()
        # poses = [pose_0.between(x.inverse()) for x in poses]
        # viz_utils.plot_poses_3d(poses, ax, center_marker_color="m", label_name="rig")
        # plt.show()

        # logger.info("Loading %d timestamps", self._number_of_timestamps_available)

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

        return ((max_num_images - 1) // self._step_size) + 1

    # def __get_timestamps(self):


    def __load_calibration(self, cam_idx: int) -> Tuple[Cal3Fisheye, Pose3]:
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
            cam_T_imu: Pose3 = self.__load_pose_relative_to_imu(calibration_data)

        return intrinsics, cam_T_imu

    def __load_intrinsics(self, calibration_data: Dict[Any, Any]) -> Cal3Fisheye:
        fx, fy, px, py = calibration_data["intrinsics"]
        k1, k2, k3, k4 = calibration_data["distortion_coeffs"]

        return Cal3Fisheye(fx=fx, fy=fy, s=0, u0=px, v0=py, k1=k1, k2=k2, k3=k3, k4=k4)

    def __load_pose_relative_to_imu(self, calibration_data: Dict[Any, Any]) -> Pose3:
        transformation_matrix: np.ndarray = calibration_data["T_cam_imu"]
        return Pose3(transformation_matrix)

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
        rig_idx_for_cam = self.__map_image_idx_to_rig(index)

        logger.debug("Mapping %d index to image %d of camera %d", index, rig_idx_for_cam, cam_for_index)

        camera_folder: Path = self.__get_folder_for_images(cam_for_index)
        image_path: Path = camera_folder / f"left{rig_idx_for_cam:04}.jpg"

        return io_utils.load_image(str(image_path))

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3Fisheye]:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        return self._intrinsics[self.__map_index_to_camera(index)]

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            the camera pose w_P_index.
        """
        return None

    def get_imu_pose_prior(self, index: int):
        return


    def get_relative_pose_prior(self, i1: int, i2: int) -> Optional[PosePrior]:
        rig_idx_for_i1: int = self.__map_image_idx_to_rig(i1)
        rig_idx_for_i2: int = self.__map_image_idx_to_rig(i2)

        cam_idx_for_i1: int = self.__map_index_to_camera(i1)
        cam_idx_for_i2: int = self.__map_index_to_camera(i2)

        i1_T_imu: Pose3 = self._cam_T_imu_poses[cam_idx_for_i1]
        i2_T_imu: Pose3 = self._cam_T_imu_poses[cam_idx_for_i2]

        if rig_idx_for_i1 == rig_idx_for_i2:
            i2Ti1 = i2_T_imu.inverse().between(i1_T_imu.inverse())
            return PosePrior(value=i2Ti1, covariance=None, type=PosePriorType.HARD_CONSTRAINT)
        else:
            #TODO: Jon add relative PosePrior for images from different timestamps i.e. different rig_idx
            #caluculate pose from one imu to another
            imu1 = self.get_absolute_pose(i1).value
            imu2 = self.get_absolute_pose(i2).value

            return None

    #TODO: Jon
    def get_absolute_pose(self, i1: int): -> Optional[Pose3]:
        timestamp = self.__map_image_idx_to_timestamp(i1)
        pose1 = self.timestamp_to_pose[timestamp]
        return pose1



    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair. idx1 < idx2 is required.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """
        return super().is_valid_pair(idx1, idx2) and abs(idx1 - idx2) <= self._max_frame_lookahead

    def store_fastlio_output(self):
        with open(self._fastlio_output_path) as file:
            lines = file.readlines()
            timestamp_to_pose = {}
            for line in lines:
                line = line.split()
                timestamp = line[0]
                t = np.asarray(line[1:4])
                quat = np.asarray(line[4:8])
                R = Rot3(np.asarray(scipyR.from_quat(quat).as_matrix()))
                timestamp_to_pose[timestamp] = Pose3(R, t)
        return timestamp_to_pose

    #TODO: Jon
    def __map_image_idx_to_timestamp(self, index: int):
        return None

    def __map_index_to_camera(self, index: int) -> int:
        return self._cams_to_use[index % len(self._cams_to_use)]

    def __map_image_idx_to_rig(self, index: int) -> int:
        return index // len(self._cams_to_use) * self._step_size


if __name__ == "__main__":
    root = "/Users/jonwomack/Documents/projects/swarmgt/experiments/hilti/"
    print(root)

    loader = HiltiLoader(root, {0, 1, 2, 3, 4})

    # T_cn_cnm1 = np.array(
    #     [
    #         [0.9999761426988956, 0.004698109260098521, -0.005063773535634404, 0.10817339479208792],
    #         [-0.004705552944957792, -0.9999878643586437, -0.0014590774217762541, -0.0005128409082424196],
    #         [0.005056857178348414, 0.0014828704666000705, 0.9999861145489266, 0.0006919599620310546],
    #         [0, 0, 0, 1],
    #     ]
    # )
    #
    # c1Tc0 = Pose3(T_cn_cnm1)
    # print(c1Tc0.rotation().xyz())
    # print(c1Tc0.translation())
    #
    # pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (3, 4)]
    # for i1, i2 in pairs:
    #     pose = loader.get_relative_pose_prior(i1, i2).value
    #     print(i1, i2)
    #     print(pose.rotation().xyz())
    #     print(pose.translation())
    #
    # # for i in range(100):
    # #     loader.get_image(i)
