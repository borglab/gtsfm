"""MVS math methods for gtsfm

Authors: Ren Liu
"""
import math
import numpy as np


def angle_between_vectors(v_a: np.ndarray, v_b: np.ndarray) -> float:
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
    theta = angle_between_vectors(p_a, p_b)
    if theta <= theta_0:
        return math.exp(-((theta - theta_0) ** 2) / (2 * sigma_1 ** 2))
    else:
        return math.exp(-((theta - theta_0) ** 2) / (2 * sigma_2 ** 2))


def to_camera_coordinates(p: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    """convert world coordinates to camera coordinates

    Args:
        p: pose vector in np.ndarray of [3, ] shape,
        camera_pose: target camera pose, a 4x4 np.ndarray

    Returns:
        pose vector in np.ndarray of [3, ] shape in target camera perspective
    """
    homo_p = np.ones([4])
    homo_p[:3] = p
    cam_p = camera_pose @ homo_p.reshape([4, 1])
    cam_p /= cam_p[3, 0]

    return cam_p.reshape([4])[:3]
