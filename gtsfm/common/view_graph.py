"""Defines the ViewGraph data structure.

Authors: Akshay Krishnan
"""
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from gtsam import Rot3, Unit3, Cal3Bundler


@dataclass
class ViewGraph:
    """A data structure that stores all the two-view relationships between cameras in the scene.

    Args:
        i2Ri1: dictionary of relative rotations between camera pairs.
        i2Ui1: dictionary of relative unit-translations between camera pairs.
        calibrations: list of calibrations for each camera.
        corr_idxs_i1i2: dictionary of correspondence indices between camera pairs.
    """

    i2Ri1: Dict[Tuple[int, int], Rot3]
    i2Ui1: Dict[Tuple[int, int], Unit3]
    calibrations: List[Cal3Bundler]
    corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray]

    def get_pair_indices(self) -> List[Tuple[int, int]]:
        """Get the indices of the camera pairs, which are the edges of this graph."""
        # TODO: assert the edges are the same in all three dicts.
        return list(self.i2Ri1.keys())
