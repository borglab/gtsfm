"""Utilities for the multi-view stereo (MVS) stage of the back-end.

Authors: Ren Liu
"""
import math


def piecewise_gaussian(theta: float, theta_0: float = 5, sigma_1: float = 1, sigma_2: float = 10) -> float:
    """A Gaussian function that favors a certain baseline angle (theta_0).

    The Gaussian function is divided into two pieces with different standard deviations:
    - If the input baseline angle (theta) is no larger than theta_0, the standard deviation will be sigma_1;
    - If theta is larger than theta_0, the standard deviation will be sigma_2.

    More details can be found in "View Selection" paragraphs in Yao's paper https://arxiv.org/abs/1804.02505.

    Args:
        theta: the input baseline angle.
        theta_0: defaults to 5, the expected baseline angle of the function.
        sigma_1: defaults to 1, the standard deviation of the function when the angle is no larger than theta_0.
        sigma_2: defaults to 10, the standard deviation of the function when the angle is larger than theta_0.

    Returns:
        the result of the Gaussian function, in range (0, 1]
    """

    # Decide the standard deviation according to theta and theta_0
    # if theta is no larger than theta_0, the standard deviation is sigma_1
    if theta <= theta_0:
        sigma = sigma_1
    # if theta is larger than theta_0, the standard deviation is sigma_2
    else:
        sigma = sigma_2

    return math.exp(-((theta - theta_0) ** 2) / (2 * sigma ** 2))
