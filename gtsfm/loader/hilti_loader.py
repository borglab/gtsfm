"""Hilti dataset loader.

The dataset should be preprocessed to extract images from each camera into its respective folders.

Dataset ref: https://rpg.ifi.uzh.ch/docs/Arxiv21_HILTI.pdf
Kalibr format for intrinsics: https://github.com/ethz-asl/kalibr/wiki/yaml-formats
Data extracted from rosbag using section 0.1 of http://wiki.ros.org/rosbag/Tutorials/Exporting%20image%20and%20video%20data

Authors: Ayush Baid
"""
import glob
import yaml
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

import cv2
import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Fisheye, Pose3

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

INTRA_RIG_VALID_PAIRS = {(0, 1), (0, 3), (1, 4)}
INTER_RIG_VALID_PAIRS = {(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 4), (2, 2), (3, 0), (3, 3), (4, 1), (4, 4)}


class HiltiLoader(LoaderBase):
    def __init__(
        self,
        base_folder: str,
        cams_to_use: Set[int],
        max_resolution: int = 1000,
        max_frame_lookahead: int = 10,
        step_size: int = 8,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__(max_resolution=max_resolution)
        if step_size % 4 != 0:
            raise ValueError("Step size must be multiple of 4 to match lidar frequency")
        self.__check_cams_to_use(cams_to_use)
        self._base_folder: Path = Path(base_folder)
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
        if self._max_length is not None:
            self._number_of_timestamps_available = min(self._number_of_timestamps_available, self._max_length)

        # import matplotlib.pyplot as plt
        # import gtsfm.utils.viz as viz_utils

        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")

        # poses = [self._cam_T_imu_poses[i] for i in self._cams_to_use]
        # pose_0 = poses[0].inverse()
        # poses = [pose_0.between(x.inverse()) for x in poses]
        # viz_utils.plot_poses_3d(poses, ax, center_marker_color="m", label_name="rig")
        # plt.show()

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

        return ((max_num_images - 1) // self._step_size) + 1

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

    def get_image(self, index: int) -> Image:
        return self.get_image_full_res(index)

    def get_image_undistorted(self, index: int) -> Image:
        distorted_image: Image = self.get_image(index)
        calibration: Cal3Fisheye = self.get_camera_intrinsics(index)

        new_image_size = (1500, 1500)
        Knew = calibration.K()
        Knew[0, 2] = 750
        Knew[1, 2] = 750

        undistorted_image_array: np.ndarray = cv2.fisheye.undistortImage(
            distorted_image.value_array,
            calibration.K(),
            np.array([calibration.k1(), calibration.k2(), calibration.k3(), calibration.k4()]),
            Knew=Knew,
            new_size=new_image_size,
        )

        return Image(value_array=undistorted_image_array, exif_data={})

    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            Image: the image at the query index.
        """
        cam_idx = self.__map_index_to_camera(index)
        rig_idx = self.__map_image_idx_to_rig(index)

        logger.debug("Mapping %d index to rig %d,  camera %d", index, rig_idx, cam_idx)

        camera_folder: Path = self.__get_folder_for_images(cam_idx)
        image_path: Path = camera_folder / f"left{rig_idx:04}.jpg"

        return io_utils.load_image(str(image_path))

    def get_camera_intrinsics(self, index: int) -> Optional[Cal3Fisheye]:
        return self.get_camera_intrinsics_full_res(index)

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

    def get_relative_pose_prior(self, i1: int, i2: int) -> Optional[PosePrior]:
        rig_idx_for_i1: int = self.__map_image_idx_to_rig(i1)
        rig_idx_for_i2: int = self.__map_image_idx_to_rig(i2)

        if rig_idx_for_i1 == rig_idx_for_i2:
            cam_idx_for_i1: int = self.__map_index_to_camera(i1)
            cam_idx_for_i2: int = self.__map_index_to_camera(i2)

            i1_T_imu: Pose3 = self._cam_T_imu_poses[cam_idx_for_i1]
            i2_T_imu: Pose3 = self._cam_T_imu_poses[cam_idx_for_i2]

            i2Ti1 = i2_T_imu.inverse().between(i1_T_imu.inverse())

            return PosePrior(value=i2Ti1, covariance=None, type=PosePriorType.HARD_CONSTRAINT)
        else:
            # TODO(jon): read from lidar
            return None

    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair. idx1 < idx2 is required.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """
        if not super().is_valid_pair(idx1, idx2):
            return False

        rig_idx_i1 = self.__map_image_idx_to_rig(idx1)
        rig_idx_i2 = self.__map_image_idx_to_rig(idx2)

        cam_idx_i1 = self.__map_index_to_camera(idx1)
        cam_idx_i2 = self.__map_index_to_camera(idx2)
        if rig_idx_i1 == rig_idx_i2:
            return (cam_idx_i1, cam_idx_i2) in INTRA_RIG_VALID_PAIRS
        elif rig_idx_i1 < rig_idx_i2 and rig_idx_i2 - rig_idx_i1 <= self._max_frame_lookahead * self._step_size:
            return (cam_idx_i1, cam_idx_i2) in INTER_RIG_VALID_PAIRS

    def __map_index_to_camera(self, index: int) -> int:
        return self._cams_to_use[index % len(self._cams_to_use)]

    def __map_image_idx_to_rig(self, index: int) -> int:
        return index // len(self._cams_to_use) * self._step_size

    def create_computation_graph_for_relative_pose_priors(self) -> Dict[Tuple[int, int], Delayed]:
        pairs = set(self.get_valid_pairs())
        # just add all possible pairs which belong to the same rig (as it will have hard relative prior)
        for i in range(len(self)):
            for j in range(i + 1, i + len(self._cams_to_use) - 1):
                if self.__map_image_idx_to_rig(i) == self.__map_image_idx_to_rig(j):
                    pairs.add((i, j))

        return {(i1, i2): dask.delayed(self.get_relative_pose_prior)(i1, i2) for i1, i2 in pairs}


if __name__ == "__main__":
    root = "/media/ayush/cross_os1/dataset/hilti"

    loader = HiltiLoader(root, {0, 1, 2, 3, 4})

    # T_cn_cnm1 = np.array(
    #     [
    #         [0.9999761426988956, 0.004698109260098521, -0.005063773535634404, 0.10817339479208792],
    #         [-0.004705552944957792, -0.9999878643586437, -0.0014590774217762541, -0.0005128409082424196],
    #         [0.005056857178348414, 0.0014828704666000705, 0.9999861145489266, 0.0006919599620310546],
    #         [0, 0, 0, 1],
    #     ]
    # )

    # c1Tc0 = Pose3(T_cn_cnm1)
    # print(c1Tc0.rotation().xyz())
    # print(c1Tc0.translation())

    # pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (3, 4)]
    # for i1, i2 in pairs:
    #     pose = loader.get_relative_pose_prior(i1, i2).value
    #     print(i1, i2)
    #     print(pose.rotation().xyz())
    #     print(pose.translation())

    # for i in range(100):
    # import gtsfm.utils.io as io_utils

    for i in range(5):
        distorted_image = loader.get_image(i)
        undistorted_image = loader.get_image_undistorted(i)

        io_utils.save_image(distorted_image, f"/home/ayush/hilti_updates/{i}_distorted.jpg")
        io_utils.save_image(undistorted_image, f"/home/ayush/hilti_updates/{i}_undistorted.jpg")
