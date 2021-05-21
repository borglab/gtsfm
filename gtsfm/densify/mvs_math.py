"""MVS math methods for gtsfm

Authors: Ren Liu
"""
import math
import numpy as np

from gtsfm.utils.geometry_comparisons import angle_between_vectors


def piecewise_gaussian(
    p_a: np.ndarray, p_b: np.ndarray, theta_0: float = 5, sigma_1: float = 1, sigma_2: float = 10
) -> float:
    """Evaluate the similarity of measurements for the same track p in view a and view b.

    This piecewise Gaussian function will outputs a float score to show the evaluation result of the similarity.
    The total similarity between two views can be calcualted by summing up the scores of all common tracks.

    Details can be seen in "View Selection" paragraphs in Yao's paper https://arxiv.org/abs/1804.02505.

    Args:
        p_a: measurement of track p in view a in shape of [3, ],
        p_b: measurement of track p in view b in shape of [3, ],
        theta_0: threshold angle between measurements in different views, the default threshold is 5 degree.
        sigma_1: if the angle between measurements in different views is no larger than the threshold angle, which means
            view a and b are similar in this track, then the gaussian variance should be smaller to make the score
            higher. The default sigma_1 is set to be 1 according to the paper.
        sigma_2: if the angle between measurements in different views is larger than the threshold angle, which means
            view a and b are not similar in this track, so less importance should be attached to this track. The
            gaussian variance should be larger to make the score lower. The default sigma_2 is set to be 10.

    Returns:
        float piecewice gaussian value as the score of the track between two views
    """
    # 1. calculate the angle between measurement poses of track p in views a and b.
    theta = angle_between_vectors(p_a, p_b)
    # 2. calculate the score according to the angle
    if theta <= theta_0:  # if the angle is no larger than the thresold, we should attach more importance
        return math.exp(-((theta - theta_0) ** 2) / (2 * sigma_1 ** 2))
    else:  # if the angle is larger than the thresold, we should attach less importance
        return math.exp(-((theta - theta_0) ** 2) / (2 * sigma_2 ** 2))
