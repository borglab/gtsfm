"""Datastructure for pose prior

Authors: Ayush Baid
"""
from typing import NamedTuple

import numpy as np
from gtsam import Pose3


class PosePriorType(str, Enum):
    HARD_CONSTRAINT = "hard_constraint"
    SOFT_CONSTRAINT = "soft_constraint"


class PosePrior(NamedTuple):
    value: Pose3
    covariance: np.ndarray
    type: PosePriorType
