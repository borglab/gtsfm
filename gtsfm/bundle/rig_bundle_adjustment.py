"""Special Bundle adjustment optimizer to handle rigs: devices with multiple sensors.

Right now, this class is specific to the rig in the Hilti 2022 challenge, and needs generalization.
TODO(Ayush): generalize.

Authors: Ayush Baid.
"""
from typing import Dict, List, Optional, Tuple

import gtsam
from gtsam import BetweenFactorPose3, NonlinearFactorGraph

import gtsfm.utils.logger as logger_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer, X
from gtsfm.common.pose_prior import PosePrior


logger = logger_utils.get_logger()


class RigBundleAdjustmentOptimizer(BundleAdjustmentOptimizer):
    """Bundle adjustment for special rig multi-sensory devices."""

    # TODO(Ayush): share calibrations between same cams?
    # TODO(Ayush): use a diagonal model for calibration prior as distortion should have much lower sigmas?

    def __get_rig_idx(self, camera_idx: int) -> int:
        return camera_idx // 5

    def __get_camera_type(self, camera_idx: int) -> int:
        return camera_idx % 5

    def _between_factors(
        self, relative_pose_priors: Dict[Tuple[int, int], Optional[PosePrior]], cameras_to_model: List[int]
    ) -> NonlinearFactorGraph:
        """Add between factors on poses between cameras and IMUs.

        1. For the same timestamp, add a prior factor between each camera and cam2.
        2. For different timestamps, the between factors are between cam2s only.
        """
        graph = NonlinearFactorGraph()

        # translate the relative pose priors between cams to IMUs, and add if not already present
        between_factors: Dict[Tuple[int, int], BetweenFactorPose3] = {}
        for (i1, i2), i2Ti1_prior in relative_pose_priors.items():
            if i2Ti1_prior is None or i1 not in cameras_to_model or i2 not in cameras_to_model:
                continue

            b1: int = self.__get_rig_idx(i1)
            b2: int = self.__get_rig_idx(i2)
            cam_type_i1: int = self.__get_camera_type(i1)
            cam_type_i2: int = self.__get_camera_type(i2)

            if (b1 == b2 and (cam_type_i1 == 2 or cam_type_i2 == 2)) or (cam_type_i1 == 2 and cam_type_i2 == 2):
                between_factors[(i1, i2)] = BetweenFactorPose3(
                    X(i2), X(i1), i2Ti1_prior.value, gtsam.noiseModel.Diagonal.Sigmas(i2Ti1_prior.covariance)
                )

        logger.info("Added %d between factors for BA", len(between_factors))

        for factor in between_factors.values():
            graph.push_back(factor)

        return graph
