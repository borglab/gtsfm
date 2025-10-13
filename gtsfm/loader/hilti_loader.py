"""Hilti dataset loader.

The dataset should be preprocessed to extract images in sync with the lidar information.

Folder structure:
- images/*.jpg: contains images from the 5 cameras with the following naming convention: 0-4 from the 0th lidar
    timestamp, 5-9 from the 1st, and so on.
- calibration/: contains the calibration data downloaded from Hilti's official website
- lidar/ contains files fastlio2.g2o and fastlio2_odom.txt from the SLAM.

Dataset ref: https://rpg.ifi.uzh.ch/docs/Arxiv21_HILTI.pdf
Kalibr format for intrinsics: https://github.com/ethz-asl/kalibr/wiki/yaml-formats

Authors: Ayush Baid
"""

import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gtsam  # type: ignore
import numpy as np
import yaml  # type: ignore
from gtsam import Cal3Fisheye, Pose3

import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.constraint import Constraint
from gtsfm.common.image import Image
from gtsfm.common.pose_prior import PosePrior, PosePriorType
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.products.visibility_graph import VisibilityGraph

logger = logger_utils.get_logger()

NUM_CAMS = 5

CAM_IDX_TO_KALIBR_FILE_MAP = {
    0: "calib_3_cam0-1-camchain-imucam.yaml",
    1: "calib_3_cam0-1-camchain-imucam.yaml",
    2: "calib_3_cam2-camchain-imucam.yaml",
    3: "calib_3_cam3-camchain-imucam.yaml",
    4: "calib_3_cam4-camchain-imucam.yaml",
}

LIDAR_POSE_RELATIVE_PATH = "lidar/fastlio2.g2o"
LIDAR_CONSTRAINTS_RELATIVE_PATH = "lidar/constraints.txt"
IMAGES_FOLDER = "images"

HARD_RELATIVE_POSE_PRIOR_SIGMA = np.ones((6,)) * 1e-3  # CAM_IMU_POSE_PRIOR_SIGMA in BA should have similar value
SOFT_RELATIVE_POSE_PRIOR_SIGMA = np.ones((6,)) * 3e-2
SOFT_ABSOLUTE_POSE_PRIOR_SIGMA = np.ones((6,)) * 3e-2


class HiltiLoader(LoaderBase):
    def __init__(
        self,
        dataset_dir: str,
        images_dir: Optional[str] = None,
        max_length: Optional[int] = None,
        max_resolution: int = 1080,
        input_worker: Optional[str] = None,
    ) -> None:
        """Initializes, loads calibration, constraints, and pose priors.

        Args:
            dataset_dir (str): top-level dataset directory, expects calibration, images and lidar sub-folders.
            images_dir (Optional[str]): path to images directory. If None, defaults to {dataset_dir}/images.
            max_length (Optional[int]): limit poses to read. Defaults to None.
            max_resolution: integer representing maximum length of image's short side
               e.g. for 1080p (1920 x 1080), max_resolution would be 1080
        """
        super().__init__(max_resolution, input_worker)
        self._dataset_dir: Path = Path(dataset_dir)
        self._images_dir: Path = Path(images_dir) if images_dir else self._dataset_dir / IMAGES_FOLDER
        self._max_length = max_length

        # Load calibration.
        self._intrinsics: Dict[int, Cal3Fisheye] = {}
        self._cam_T_imu_poses: Dict[int, Pose3] = {}
        for cam_idx in range(NUM_CAMS):
            calibration = self.__load_calibration(cam_idx)
            self._intrinsics[cam_idx] = calibration[0]
            self._cam_T_imu_poses[cam_idx] = calibration[1]

        # Check how many images are on disk.
        self.num_rig_poses: int = self.__get_num_rig_poses()
        if self._max_length is not None:
            self.num_rig_poses = min(self.num_rig_poses, self._max_length)

        # Read the constraints from the lidar/constraints file
        self.constraints = self.__load_constraints()
        logger.info("Number of constraints: %d", len(self.constraints))

        # Read the poses for the IMU for rig indices from g2o file.
        self._w_T_imu: Dict[int, Pose3] = self.__read_lidar_pose_priors()

        logger.info("Loading %d timestamps", self.num_rig_poses)
        logger.info("Lidar camera available for %d timestamps", len(self._w_T_imu))

    def __load_constraints(self) -> List[Constraint]:
        constraints_path = self._dataset_dir / LIDAR_CONSTRAINTS_RELATIVE_PATH
        constraints = Constraint.read(str(constraints_path))

        # filter them according to max length
        constraints = list(filter(lambda c: c.a < self.num_rig_poses and c.b < self.num_rig_poses, constraints))

        return constraints

    def get_cam_T_imu(self) -> Dict[int, Pose3]:
        return self._cam_T_imu_poses

    def __read_lidar_pose_priors(self) -> Dict[int, Pose3]:
        """Read the poses for the IMU for rig indices."""
        filepath = str(self._dataset_dir / LIDAR_POSE_RELATIVE_PATH)
        _, values = gtsam.readG2o(filepath, is3D=True)

        lidar_keys = values.keys()
        logger.info("Number of keys in g2o file: %d", len(lidar_keys))

        w_T_imu: Dict[int, Pose3] = {}

        for rig_idx in range(self.num_rig_poses):
            if rig_idx in lidar_keys:
                w_T_imu[rig_idx] = values.atPose3(rig_idx)

        return w_T_imu

    def __get_num_rig_poses(self) -> int:
        """Check how many images we have on disk and deduce number of rig poses."""
        search_path: str = str(self._images_dir / "*.jpg")
        image_files = glob.glob(search_path)
        total_num_images = len(image_files)
        return total_num_images // NUM_CAMS

    def __load_calibration(self, cam_idx: int) -> Tuple[Cal3Fisheye, Pose3]:
        """Load calibration from kalibr files in calibration sub-folder."""
        kalibr_file_path = self._dataset_dir / "calibration" / CAM_IDX_TO_KALIBR_FILE_MAP[cam_idx]

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
        """Create gtsam.Cal3Fisheye object from calibration data."""
        fx, fy, px, py = calibration_data["intrinsics"]
        k1, k2, k3, k4 = calibration_data["distortion_coeffs"]

        return Cal3Fisheye(fx=fx, fy=fy, s=0, u0=px, v0=py, k1=k1, k2=k2, k3=k3, k4=k4)

    def __load_pose_relative_to_imu(self, calibration_data: Dict[Any, Any]) -> Pose3:
        """Create gtsam.Pose3 object from calibration data"""
        transformation_matrix: np.ndarray = calibration_data["T_cam_imu"]
        return Pose3(transformation_matrix)

    def __len__(self) -> int:
        """
        The number of images in the dataset.

        Returns:
            the number of images.
        """
        return self.num_rig_poses * NUM_CAMS

    def get_image(self, index: int) -> Image:
        return self.get_image_full_res(index)

    # def get_image_undistorted(self, index: int) -> Image:
    #     distorted_image: Image = self.get_image(index)
    #     calibration: Cal3Fisheye = self.get_camera_intrinsics(index)

    #     new_image_size = (1500, 1500)
    #     Knew = calibration.K()
    #     Knew[0, 2] = 750
    #     Knew[1, 2] = 750

    #     undistorted_image_array: np.ndarray = cv2.fisheye.undistortImage(
    #         distorted_image.value_array,
    #         calibration.K(),
    #         np.array([calibration.k1(), calibration.k2(), calibration.k3(), calibration.k4()]),
    #         Knew=Knew,
    #         new_size=new_image_size,
    #     )

    #     return Image(value_array=undistorted_image_array, exif_data={})

    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            Image: the image at the query index.
        """
        image_path: Path = self._images_dir / f"{index}.jpg"

        return io_utils.load_image(str(image_path))

    def image_filenames(self) -> List[str]:
        """Return the file names corresponding to each image index."""
        return [f"{i}.jpg" for i in range(len(self))]

    def get_camera_intrinsics(self, index: int) -> Optional[Cal3Fisheye]:
        return self.get_camera_intrinsics_full_res(index)

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3Fisheye]:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        return self._intrinsics[self.camera_from_image(index)]

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Note: temporarily using the lidar poses as ground truth

        Args:
            index: the index to fetch.

        Returns:
            the camera pose w_P_index.
        """
        rig_idx: int = self.rig_from_image(index)
        cam_idx: int = self.camera_from_image(index)

        if rig_idx in self._w_T_imu:
            return self._w_T_imu[rig_idx] * self._cam_T_imu_poses[cam_idx].inverse()

        return None

    def get_relative_pose_prior(self, i1: int, i2: int) -> Optional[PosePrior]:
        """Creates prior on relative pose i2Ti1.

        If the images are on the same rig, then creates a hard pose prior between the two images, to be used by the
        two-view estimator. If they are not, we create a soft pose prior derived from the absolute poses (w_T_imu)
        passed in the constructor.

        Args:
            i1: index of first image.
            i2: index of second image.

        Returns:
            Pose prior, if it exists.
        """
        rig_idx_for_i1: int = self.rig_from_image(i1)
        rig_idx_for_i2: int = self.rig_from_image(i2)
        cam_idx_for_i1: int = self.camera_from_image(i1)
        cam_idx_for_i2: int = self.camera_from_image(i2)

        # TODO(Frank): this should come from constraints?

        if rig_idx_for_i1 == rig_idx_for_i2:
            i1_T_imu: Pose3 = self._cam_T_imu_poses[cam_idx_for_i1]
            i2_T_imu: Pose3 = self._cam_T_imu_poses[cam_idx_for_i2]
            i2Ti1 = i2_T_imu.inverse().between(i1_T_imu.inverse())
            # TODO: add covariance
            return PosePrior(value=i2Ti1, covariance=HARD_RELATIVE_POSE_PRIOR_SIGMA, type=PosePriorType.HARD_CONSTRAINT)
        elif rig_idx_for_i1 in self._w_T_imu and rig_idx_for_i2 in self._w_T_imu:
            w_T_i1 = self._w_T_imu[rig_idx_for_i1] * self._cam_T_imu_poses[cam_idx_for_i1].inverse()
            w_T_i2 = self._w_T_imu[rig_idx_for_i2] * self._cam_T_imu_poses[cam_idx_for_i2].inverse()
            i2Ti1 = w_T_i2.between(w_T_i1)
            # TODO: add covariance
            return PosePrior(value=i2Ti1, covariance=SOFT_RELATIVE_POSE_PRIOR_SIGMA, type=PosePriorType.SOFT_CONSTRAINT)

        return None

    def get_absolute_pose_prior(self, idx: int) -> Optional[PosePrior]:
        rig_idx: int = self.rig_from_image(idx)
        cam_idx: int = self.camera_from_image(idx)

        if rig_idx in self._w_T_imu:
            w_T_cam = self._w_T_imu[rig_idx] * self._cam_T_imu_poses[cam_idx].inverse()
            return PosePrior(
                value=w_T_cam, covariance=SOFT_ABSOLUTE_POSE_PRIOR_SIGMA, type=PosePriorType.SOFT_CONSTRAINT
            )

        return None

    def camera_from_image(self, index: int) -> int:
        """Map image index to camera-on-rig index."""
        return index % NUM_CAMS

    def rig_from_image(self, index: int) -> int:
        """Map image index to rig index."""
        return index // NUM_CAMS

    def image_from_rig_and_camera(self, rig_index: int, camera_idx: int) -> int:
        """Map image index to rig index."""
        return rig_index * NUM_CAMS + camera_idx

    def get_relative_pose_priors(self, visibility_graph: VisibilityGraph) -> Dict[Tuple[int, int], PosePrior]:
        pairs = set(visibility_graph)
        # For every rig index, add a "star" from camera 2 to 0,1,3,4:
        for rig_index in range(self.num_rig_poses):
            camera_2 = self.image_from_rig_and_camera(rig_index, 2)
            for cam_idx in [0, 1, 3, 4]:
                pairs.add((camera_2, self.image_from_rig_and_camera(rig_index, cam_idx)))

        maybe_priors = {pair: self.get_relative_pose_prior(*pair) for pair in pairs}
        priors = {pair: prior for pair, prior in maybe_priors.items() if prior is not None}

        return priors
