"""Datastructure for pose prior

Authors: Ayush Baid
"""
from enum import Enum
from typing import NamedTuple

import numpy as np
from gtsam import Pose3


class PosePriorType(str, Enum):
    HARD_CONSTRAINT = "hard_constraint"
    SOFT_CONSTRAINT = "soft_constraint"


class PosePrior(NamedTuple):
    # prior values for the camera poses
    value: Pose3
    covariance: np.ndarray  # its actually sigma
    type: PosePriorType
