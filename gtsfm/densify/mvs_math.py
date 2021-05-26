"""MVS math methods for gtsfm

Authors: Ren Liu
"""
import math
import numpy as np

from gtsfm.utils.geometry_comparisons import angle_between_vectors


def piecewise_gaussian(
    a_x: np.ndarray, b_x: np.ndarray, theta_0: float = 5, sigma_1: float = 1, sigma_2: float = 10
) -> float:
    """Evaluate the similarity of measurements for the same track in camera a and camera b.

    This piecewise Gaussian function outputs a float score to show the evaluation result of the similarity.
    The total similarity between two views can be calculated by summing up the scores of all common tracks.

    Details can be seen in "View Selection" paragraphs in Yao's paper https://arxiv.org/abs/1804.02505.

    Args:
        a_x: 3D coordinates of the track point in pose a, with shape (3,),
        b_x: 3D coordinates of the track point in pose b, with shape (3,),
        theta_0: Default theta_0 is set to be 5.
            theta_0 is the threshold angle (in degree) between coordinates in different poses.
        sigma_1: Default sigma_1 is set to be 1.
            If the angle between measurements in different views is no larger than the threshold angle, which means
            view a and b are similar in this track, then the gaussian variance should be smaller to make the score
            higher.
        sigma_2: Default sigma_2 is set to be 10.
            If the angle between measurements in different views is larger than the threshold angle, which means
            view a and b are not similar in this track, so less importance should be attached to this track. The
            gaussian variance should be larger to make the score lower.

    Returns:
        A score of the track between two views in the range (0,1]
    """
    # 1. calculate the angle between measurement poses of track p in views a and b.
    theta = angle_between_vectors(a_x, b_x)
    # 2. calculate the score according to the angle
    if theta <= theta_0:  # if the angle is no larger than the threshold, we should attach more importance
        return math.exp(-((theta - theta_0) ** 2) / (2 * sigma_1 ** 2))
    else:  # if the angle is larger than the threshold, we should attach less importance
        return math.exp(-((theta - theta_0) ** 2) / (2 * sigma_2 ** 2))
