"""Special Bundle adjustment optimizer to handle rigs: devices with multiple sensors.

Right now, this class is specific to the rig in the Hilti 2022 challenge, and needs generalization. The hilti dataset
has 5 cameras which are fired in sync. Among the 5 cameras, camera #2 is the one facing up and is used to add intra-rig
and inter-rig between factors.
TODO(Ayush): generalize.

Authors: Ayush Baid.
"""
from typing import Dict, List, Tuple

import gtsam
from gtsam import BetweenFactorPose3, NonlinearFactorGraph

import gtsfm.utils.logger as logger_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer, X
from gtsfm.common.pose_prior import PosePrior

UPWARD_FACING_CAM_TYPE = 2


logger = logger_utils.get_logger()


class RigBundleAdjustmentOptimizer(BundleAdjustmentOptimizer):
    """Bundle adjustment for special rig multi-sensory devices."""

    # TODO(Ayush): share calibrations between same cams?
    # TODO(Ayush): use a diagonal model for calibration prior as distortion should have much lower sigmas?

    def __get_rig_idx(self, camera_idx: int) -> int:
        """Get the rig index pertaining to the camera, as there are 5 cameras in sync per timestamp."""
        return camera_idx // 5

    def __get_camera_type(self, camera_idx: int) -> int:
        """Get the type of the camera, i.e. the indexing of the camera according to its physical location in the rig."""
        return camera_idx % 5

    def _between_factors(
        self, relative_pose_priors: Dict[Tuple[int, int], PosePrior], cameras_to_model: List[int]
    ) -> NonlinearFactorGraph:
        """Add between factors on poses between cameras and IMUs.

        1. For the same timestamp, add a prior factor between each camera and cam2.
        2. For different timestamps, the between factors are between cam2s only.
        """
        graph = NonlinearFactorGraph()

        # translate the relative pose priors between cams to IMUs, and add if not already present
        between_factors: Dict[Tuple[int, int], BetweenFactorPose3] = {}
        for (i1, i2), i2Ti1_prior in relative_pose_priors.items():
            if i1 not in cameras_to_model or i2 not in cameras_to_model:
                continue

            b1: int = self.__get_rig_idx(i1)
            b2: int = self.__get_rig_idx(i2)
            cam_type_i1: int = self.__get_camera_type(i1)
            cam_type_i2: int = self.__get_camera_type(i2)

            if (b1 == b2 and (cam_type_i1 == UPWARD_FACING_CAM_TYPE or cam_type_i2 == UPWARD_FACING_CAM_TYPE)) or (
                cam_type_i1 == UPWARD_FACING_CAM_TYPE and cam_type_i2 == UPWARD_FACING_CAM_TYPE
            ):
                between_factors[(i1, i2)] = BetweenFactorPose3(
                    X(i2), X(i1), i2Ti1_prior.value, gtsam.noiseModel.Diagonal.Sigmas(i2Ti1_prior.covariance)
                )

        logger.info("Added %d between factors for BA", len(between_factors))

        for factor in between_factors.values():
            graph.push_back(factor)

        return graph
