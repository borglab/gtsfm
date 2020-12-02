"""Shonan Rotation Averaging.

The algorithm was proposed in "Shonan Rotation Averaging:Global Optimality by
Surfing SO(p)^n" and is implemented by wrapping up over implementation provided
by GTSAM.

References:
- https://arxiv.org/abs/2008.02737
- https://gtsam.org/

Authors: Jing Wu, Ayush Baid
"""
from typing import Dict, List, Optional, Tuple

import gtsam
import numpy as np
from gtsam import BetweenFactorPose3, Pose3, Rot3, ShonanAveraging3

from averaging.rotation.rotation_averaging_base import RotationAveragingBase


class ShonanRotationAveraging(RotationAveragingBase):
    """Performs Shonan rotation averaging."""

    def __init__(self):
        self._p_min = 5
        self._p_max = 30

    def run(self,
            num_images: int,
            i1_R_i2_dict: Dict[Tuple[int, int], Optional[Rot3]]
            ) -> List[Optional[Rot3]]:
        """Run the rotation averaging.

        Args:
            num_images: number of poses.
            i1_R_i2_dict: relative rotations between pairs of camera poses (
                          rotation of i2^th pose in i1^th frame for various
                          pairs of (i1, i2). The pairs serve as keys of the
                          dictionary).

        Returns:
            Global rotations for each camera pose, i.e. w_R_i, as a list. The
                number of entries in the list is `num_images`. The list may
                contain `None` where the global rotation could not be computed
                (either underconstrained system or ill-constrained system).
        """
        # lm_params = gtsam.LevenbergMarquardtParams.CeresDefaults()
        # shonan_params = ShonanAveragingParameters3(lm_params)
        noise_model = gtsam.noiseModel.Unit.Create(6)

        between_factors = gtsam.BetweenFactorPose3s()

        for (i1,i2), i1Ri2 in i1_R_i2_dict.items():
            if i1Ri2 is not None:
                between_factors.append(
                    BetweenFactorPose3(
                        i1,
                        i2,
                        Pose3(i1Ri2, np.zeros(3,)),
                        noise_model
                    )
                )

        obj = ShonanAveraging3(between_factors)  # , shonan_params)

        initial = obj.initializeRandomly()
        result_values, _ = obj.run(initial, self._p_min, self._p_max)

        return [result_values.atRot3(i) for i in range(num_images)]
