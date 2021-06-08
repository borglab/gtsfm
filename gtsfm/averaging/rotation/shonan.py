"""Shonan Rotation Averaging.

The algorithm was proposed in "Shonan Rotation Averaging:Global Optimality by
Surfing SO(p)^n" and is implemented by wrapping up over implementation provided
by GTSAM.

References:
- https://arxiv.org/abs/2008.02737
- https://gtsam.org/

Authors: Jing Wu, Ayush Baid, John Lambert
"""
from typing import Dict, List, Optional, Tuple

import gtsam
import numpy as np
from gtsam import (
    BetweenFactorPose3,
    LevenbergMarquardtParams,
    Rot3,
    Pose3,
    ShonanAveraging3,
    ShonanAveragingParameters3,
)

from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase


class ShonanRotationAveraging(RotationAveragingBase):
    """Performs Shonan rotation averaging."""

    def __init__(self) -> None:
        """
        Note: `p_min` and `p_max` describe the minimum and maximum relaxation rank.
        """
        self._p_min = 5
        self._p_max = 30

    def run_w_consecutive_ordering(
        self, num_connected_nodes: int, i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]]
    ) -> List[Optional[Rot3]]:
        """Run the rotation averaging on a connected graph w/ N keys ordered consecutively [0,...,N-1].

        Note: the assumption that Shonan keys are ordered as [0,...,N-1] goes deeper than
        ShonanAveraging::NrUnknowns(), which just checks the assumption. Several other functions that construct
        matrices related to the convergence certificate also use the fact that the graph is connected and
        keys are 0..N-1. There are about 10 places where nrUnknowns is used to dimension a sparse or dense matrix.
        Modifying GTSAM would require a major philosophical overhaul, so we perform the re-ordering
        here in a sort of "wrapper". See: https://github.com/borglab/gtsam/issues/784

        Args:
            num_connected_nodes: number of unique connected nodes in the graph (can be far less than the number
                of images in the dataset)
            i2Ri1_dict: relative rotations for each image pair-edge as dictionary (i1, i2): i2Ri1.

        Returns:
            Global rotations for each **CONNECTED** camera pose, i.e. wRi, as a list. The number of entries in
                the list is num_connected_nodes`. The list may contain `None` where the global rotation could
                not be computed (either underconstrained system or ill-constrained system).
        """
        lm_params = LevenbergMarquardtParams.CeresDefaults()
        shonan_params = ShonanAveragingParameters3(lm_params)
        shonan_params.setUseHuber(False)
        shonan_params.setCertifyOptimality(True)

        noise_model = gtsam.noiseModel.Unit.Create(6)

        between_factors = gtsam.BetweenFactorPose3s()

        for (i1, i2), i2Ri1 in i2Ri1_dict.items():
            if i2Ri1 is not None:
                # ignore translation during rotation averaging
                i2Ti1 = Pose3(i2Ri1, np.zeros(3))
                between_factors.append(BetweenFactorPose3(i2, i1, i2Ti1, noise_model))

        obj = ShonanAveraging3(between_factors, shonan_params)

        initial = obj.initializeRandomly()
        result_values, _ = obj.run(initial, self._p_min, self._p_max)

        return [result_values.atRot3(i) if result_values.exists(i) else None for i in range(num_connected_nodes)]

    def run(self, num_images: int, i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]]) -> List[Optional[Rot3]]:
        """Wrapper that re-orders keys to prepare a connected graph, w/ N keys ordered [0,...,N-1].

        Args:
            num_images: number of poses.
            i2Ri1_dict: relative rotations for each image pair-edge as dictionary (i1, i2): i2Ri1.

        Returns:
            Global rotations for each camera pose, i.e. wRi, as a list. The number of entries in the list is
                `num_images`. The list may contain `None` where the global rotation could not be computed (either
                underconstrained system or ill-constrained system), or where the camera pose had no valid observation
                in the input to run().
        """
        connected_nodes = set()
        for (i1, i2) in i2Ri1_dict.keys():
            connected_nodes.add(i1)
            connected_nodes.add(i2)

        connected_nodes = sorted(list(connected_nodes))

        # given original index, get back new temporary index
        reordered_idx_map = {}
        for i in range(num_images):
            if i in connected_nodes:
                reordered_idx_map[i] = connected_nodes.index(i)

        # now, re-order all of the indices
        i2Ri1_dict_reordered = {}
        for (i1, i2), i2Ri1 in i2Ri1_dict.items():
            i1_ = reordered_idx_map[i1]
            i2_ = reordered_idx_map[i2]
            i2Ri1_dict_reordered[(i1_, i2_)] = i2Ri1

        wRi_list_subset = self.run_w_consecutive_ordering(
            num_connected_nodes=len(connected_nodes), i2Ri1_dict=i2Ri1_dict_reordered
        )

        wRi_list = [None] * num_images
        for i in range(num_images):
            if i in connected_nodes:
                i_ = reordered_idx_map[i]
                wRi_list[i] = wRi_list_subset[i_]

        return wRi_list
