"""
Library functionality for aligning two trajectories by fitting a SIM(3) transformation

Zichao Zhang, Davide Scaramuzza: A Tutorial on Quantitative Trajectory Evaluation for
Visual(-Inertial) Odometry, IEEE/RSJ Int. Conf. Intell. Robot. Syst. (IROS), 2018.

Code adapted from https://github.com/uzh-rpg/rpg_trajectory_evaluation
"""

from typing import Tuple

import numpy as np


def align_umeyama(
    model: np.ndarray, data: np.ndarray, known_scale: bool = False
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.
    model = s * R * data + t

    Ref: rpg_trajectory_evaluation/src/rpg_trajectory_evaluation/align_trajectory.py

    Args:
        model: Array of shape (N,3) representing first trajectory as ground truth
        data: Array of shape (N,3), representing second trajectory

    Returns:
        s: float scalar representing scale factor
        R: rotation matrix of shape (3,3)
        t: translation vector of shape (3,1)
    """

    # substract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = model.shape[0]

    # correlation
    C = 1.0 / n * np.dot(model_zerocentered.T, data_zerocentered)
    # squared l2-norm
    sigma2 = 1.0 / n * np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, Vt_svd = np.linalg.svd(C)
    D_svd = np.diag(D_svd)
    V_svd = Vt_svd.T

    S = np.eye(3)
    if np.linalg.det(U_svd) * np.linalg.det(V_svd) < 0:
        S[2, 2] = -1

    R = U_svd @ S @ V_svd.T

    if known_scale:
        s = 1
    else:
        s = 1.0 / sigma2 * np.trace(D_svd @ S)

    t = mu_M - s * np.dot(R, mu_D)

    return s, R, t
