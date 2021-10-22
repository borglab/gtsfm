"""Defines the ViewGraph data structure.

Authors: Akshay Krishnan
"""
from dataclasses import dataclass
from typing import Dict, Tuple
from dataclasses import dataclass

import numpy as np
from gtsam import Rot3, Unit3, Cal3Bundler


@dataclass
class ViewGraph:
    """A data structure that stores all the two-view relationships between cameras in the scene."""

    i2Ri1: Dict[Tuple[int, int], Rot3]
    i2Ui1: Dict[Tuple[int, int], Unit3]
    K: Dict[int, Cal3Bundler]
    correspondeces_i1_i2: Dict[Tuple[int, int], np.ndarray]
    i2Ei1: Dict[Tuple[int, int], np.ndarray] = None
    i2Fi1: Dict[Tuple[int, int], np.ndarray] = None
