"""MVS math methods for gtsfm

Authors: Ren Liu
"""
from typing import Dict, Any

import math
import numpy as np


def theta_ij(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """Calculate the angle between vector v_a and v_b

    Args:
        v_a: vector in np.ndarray of [3, ] shape
        v_b: vector in np.ndarray of [3, ] shape

    Returns:
        angle in degree with float type
    """
    return (180.0 / math.pi) * math.acos(np.dot(v_a, v_b) / np.linalg.norm(v_a) / np.linalg.norm(v_b))


def piecewise_gaussian(
    p_a: np.ndarray, p_b: np.ndarray, theta_0: float = 5, sigma_1: float = 1, sigma_2: float = 10
) -> float:
    """Calculate the piecewise gaussian value as pair distances
    reference: https://arxiv.org/abs/1804.02505

    Args:
        p_a: pose vector in np.ndarray of [3, ] shape,
        p_b: pose vector in np.ndarray of [3, ] shape,
        theta_0: float parameter,
        sigma_1: float parameter,
        sigma_2: float parameter

    Returns:
        float piecewice gaussian value
    """
    theta = theta_ij(p_a, p_b)
    if theta <= theta_0:
        return math.exp(-((theta - theta_0) ** 2) / (2 * sigma_1 ** 2))
    else:
        return math.exp(-((theta - theta_0) ** 2) / (2 * sigma_2 ** 2))
