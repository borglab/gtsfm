"""Factor-graph based formulation of Bundle adjustment and optimization.

Authors: Xiaolong Wu, John Lambert, Ayush Baid
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gtsam
from gtsam import (
    BetweenFactorPose3,
    NonlinearFactorGraph,
    Pose3,
    PriorFactorPose3,
    Values,
    symbol_shorthand,
)

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.pose_prior import PosePrior, PosePriorType
from gtsfm.loader.hilti_loader import HiltiLoader

METRICS_GROUP = "bundle_adjustment_metrics"

METRICS_PATH: Path = Path(__file__).resolve().parent.parent.parent / "result_metrics"

TEST_DATA_ROOT: Path = Path(__file__).resolve().parent.parent.parent / "tests" / "data"
HILTI_TEST_DATA_PATH: Path = TEST_DATA_ROOT / "hilti_exp4_small"

"""In this file, we use the GTSAM's GeneralSFMFactor2 instead of GeneralSFMFactor because Factor2 enables decoupling
of the camera pose and the camera intrinsics, and hence gives an option to share the intrinsics between cameras.
"""

P = symbol_shorthand.P  # 3d point
X = symbol_shorthand.X  # camera pose
K = symbol_shorthand.K  # calibration
B = symbol_shorthand.B  # body frame (IMU) pose

CAM_POSE3_DOF = 6  # 6 dof for pose of camera
CAM_CAL3BUNDLER_DOF = 3  # 3 dof for f, k1, k2 for intrinsics of camera
CAM_CAL3FISHEYE_DOF = 9
IMG_MEASUREMENT_DIM = 2  # 2d measurements (u,v) have 2 dof
POINT3_DOF = 3  # 3d points have 3 dof


# noise model params
CAM_POSE3_PRIOR_NOISE_SIGMA = 0.1
CAM_CAL3BUNDLER_PRIOR_NOISE_SIGMA = 1e-5  # essentially fixed
CAM_CAL3FISHEYE_PRIOR_NOISE_SIGMA = 1e-5  # essentially fixed
MEASUREMENT_NOISE_SIGMA = 1.0  # in pixels
HARD_POSE_PRIOR_SIGMA = 1e-3  # 1e-5 did not work as well
SOFT_POSE_PRIOR_SIGMA = 3e-2  # 1e-5 did not work as well

logger = logger_utils.get_logger()


class BundleAdjustmentHiltiOptimizer(BundleAdjustmentOptimizer):
    """Bundle adjustment using factor-graphs in GTSAM.

    This class refines global pose estimates and intrinsics of cameras, and also refines 3D point cloud structure given
    tracks from triangulation."""

    def __init__(
        self,
        output_reproj_error_thresh: Optional[float] = None,
        robust_measurement_noise: bool = False,
        shared_calib: bool = False,
        max_iterations: Optional[int] = None,
    ) -> None:
        super().__init__(output_reproj_error_thresh, robust_measurement_noise, shared_calib, max_iterations)

        # Temporary hack to get relative pose between cameras and IMUs
        hilti_loader = HiltiLoader(
            base_folder=str(HILTI_TEST_DATA_PATH),
            max_frame_lookahead=1,
            max_length=None,
        )

        self._cam_T_imu = hilti_loader.get_camTimu()

    def __get_rig_idx(self, camera_idx: int) -> int:
        return camera_idx // 5

    def __get_camera_type(self, camera_idx: int) -> int:
        return camera_idx % 5

    def _between_factors(
        self, relative_pose_priors: Dict[Tuple[int, int], Optional[PosePrior]], cameras_to_model: List[int]
    ) -> NonlinearFactorGraph:
        graph = NonlinearFactorGraph()

        # add a between factor for each camera with its IMU
        for i in cameras_to_model:
            camera_type: int = self.__get_camera_type(i)
            rig_idx: int = self.__get_rig_idx(i)
            graph.push_back(
                BetweenFactorPose3(
                    X(i),
                    B(rig_idx),
                    self._cam_T_imu[camera_type],
                    gtsam.noiseModel.Isotropic.Sigma(CAM_POSE3_DOF, HARD_POSE_PRIOR_SIGMA),
                )
            )

        # translate the relative pose priors between cams to IMUs, and add if not already present
        imu_relative_pose_priors: Dict[Tuple[int, int], BetweenFactorPose3] = {}
        for (i1, i2), i2Ti1_prior in relative_pose_priors.items():
            if (
                i2Ti1_prior is None
                or i1 not in cameras_to_model
                or i2 not in cameras_to_model
                or i2Ti1_prior.type == PosePriorType.HARD_CONSTRAINT
            ):
                continue

            b1: int = self.__get_rig_idx(i1)
            b2: int = self.__get_rig_idx(i2)

            if b1 == b2:
                # already captured in cam-IMU between factor.
                continue

            if (b1, b2) in imu_relative_pose_priors:
                continue

            i2Ti1 = i2Ti1_prior.value
            i1Tb1 = self._cam_T_imu[self.__get_camera_type(i1)]
            i2Tb2 = self._cam_T_imu[self.__get_camera_type(i2)]

            b2Tb1 = i2Tb2.inverse() * i2Ti1 * i1Tb1
            imu_relative_pose_priors[(b1, b2)] = BetweenFactorPose3(
                B(b2),
                B(b1),
                b2Tb1,
                gtsam.noiseModel.Isotropic.Sigma(CAM_POSE3_DOF, SOFT_POSE_PRIOR_SIGMA),
            )

        logger.info("Adding %d between factors for IMUs", len(imu_relative_pose_priors))

        for factor in imu_relative_pose_priors.values():
            graph.push_back(factor)

        return graph

    def _pose_priors(
        self,
        absolute_pose_priors: List[Optional[PosePrior]],
        initial_data: GtsfmData,
        camera_for_origin: gtsfm_types.CAMERA_TYPE,
    ) -> NonlinearFactorGraph:
        graph = NonlinearFactorGraph()

        # TODO(Ayush): start using absolute prior factors.
        num_priors_added = 0

        if num_priors_added == 0:
            # Adding a prior to fix origin as no absolute prior exists.
            rig_for_origin = self.__get_rig_idx(camera_for_origin)
            wTi = initial_data.get_camera(camera_for_origin).pose()
            wTb = wTi * self._cam_T_imu[self.__get_camera_type(camera_for_origin)]
            graph.push_back(
                PriorFactorPose3(
                    B(rig_for_origin),
                    wTb,
                    gtsam.noiseModel.Isotropic.Sigma(CAM_POSE3_DOF, CAM_POSE3_PRIOR_NOISE_SIGMA),
                )
            )

        return graph

    def _initial_values(self, initial_data: GtsfmData) -> Values:
        initial_values: Values = super()._initial_values(initial_data)

        # add initial values for IMU poses
        imu_poses: Dict[int, Pose3] = {}
        for i in initial_data.get_valid_camera_indices():
            camera = initial_data.get_camera(i)
            wTi = camera.pose()
            wTb = wTi * self._cam_T_imu[self.__get_camera_type(i)]
            rig_idx = self.__get_rig_idx(i)
            if rig_idx not in imu_poses:
                imu_poses[rig_idx] = wTb

        for b, wTb in imu_poses.items():
            initial_values.insert(B(b), wTb)

        return initial_values
