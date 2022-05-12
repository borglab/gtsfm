"""Shonan Rotation Averaging for the rig multi-sensory devices.

Right now, this class is specific to the Hilti 2022 challenge. The hilti dataset has 5 cameras which are fired in sync. 
Among the 5 cameras, camera #2 is the one facing up and is used to add intra-rig and inter-rig between factors. All 
other priors are ignored.

TODO(Ayush): generalize.

Authors: Ayush Baid
"""
from typing import Dict, Tuple

import gtsam
from gtsam import BetweenFactorPose3, BetweenFactorPose3s

import gtsfm.utils.logger as logger_utils
from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging
from gtsfm.common.pose_prior import PosePrior

CAM2_PRIOR_SIGMA = 0.1

UPWARD_FACING_CAM_TYPE = 2
POSE3_DOF = 6

logger = logger_utils.get_logger()


class RigShonanRotationAveraging(ShonanRotationAveraging):
    """Performs Shonan rotation averaging."""

    def __get_rig_idx(self, camera_idx: int) -> int:
        """Get the rig index pertaining to the camera, as there are 5 cameras in sync per timestamp."""
        return camera_idx // 5

    def __get_camera_type(self, camera_idx: int) -> int:
        """Get the type of the camera, i.e. the indexing of the camera according to its physical location in the rig."""
        return camera_idx % 5

    def _between_factors_from_pose_priors(
        self, i2Ti1_priors: Dict[Tuple[int, int], PosePrior], old_to_new_idxs: Dict[int, int]
    ) -> BetweenFactorPose3s:
        between_factors = BetweenFactorPose3s()

        for (i1, i2), i2Ti1_prior in i2Ti1_priors.items():
            if (
                self.__get_camera_type(i1) == UPWARD_FACING_CAM_TYPE
                or self.__get_camera_type(i2) == UPWARD_FACING_CAM_TYPE
            ):
                i2_ = old_to_new_idxs[i2]
                i1_ = old_to_new_idxs[i1]
                between_factors.append(
                    BetweenFactorPose3(
                        i2_, i1_, i2Ti1_prior.value, gtsam.noiseModel.Isotropic.Sigma(POSE3_DOF, CAM2_PRIOR_SIGMA)
                    )
                )

        return between_factors
