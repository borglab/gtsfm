"""Utility functions for computing different metrics.

Authors: Ayush Baid
"""
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Pose3, Unit3

import gtsfm.utils.features as feature_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.keypoints import Keypoints


def count_correct_correspondences(
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    intrinsics_i1: Cal3Bundler,
    intrinsics_i2: Cal3Bundler,
    i2Ti1: Pose3,
    epipolar_dist_threshold: float,
) -> int:
    """Checks the correspondences for epipolar distances and counts ones which are below the threshold.

    Args:
        keypoints_i1: keypoints in image i1.
        keypoints_i2: corr. keypoints in image i2.
        intrinsics_i1: intrinsics for i1.
        intrinsics_i2: intrinsics for i2.
        i2Ti1: relative pose
        epipolar_dist_threshold: max acceptable distance for a correct correspondence.

    Raises:
        ValueError: when the number of keypoints do not match.

    Returns:
        Number of correspondences which are correct.
    """
    # TODO: add unit test, with mocking.
    if len(keypoints_i1) != len(keypoints_i2):
        raise ValueError("Keypoints must have same counts")

    if len(keypoints_i1) == 0:
        return 0

    normalized_coords_i1 = feature_utils.normalize_coordinates(keypoints_i1.coordinates, intrinsics_i1)
    normalized_coords_i2 = feature_utils.normalize_coordinates(keypoints_i2.coordinates, intrinsics_i2)
    i2Ei1 = EssentialMatrix(i2Ti1.rotation(), Unit3(i2Ti1.translation()))

    epipolar_distances = verification_utils.compute_epipolar_distances(
        normalized_coords_i1, normalized_coords_i2, i2Ei1
    )
    return np.count_nonzero(epipolar_distances < epipolar_dist_threshold)
