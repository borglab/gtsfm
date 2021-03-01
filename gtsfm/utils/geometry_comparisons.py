"""Utility functions for comparing different types related to geometry.

Authors: Ayush Baid
"""
from typing import List, Optional, Tuple

import numpy as np
from gtsam import Point3, Pose3, Rot3, Unit3

EPSILON = np.finfo(float).eps


def align_rotations(input_list: List[Rot3], ref_list: List[Rot3]) -> List[Rot3]:
    """Aligns the list of rotations to the reference list by shifting origin.

    Args:
        input_list: input rotations which need to be aligned, suppose w1Ri in world-1 frame for all frames i.
        ref_list: reference rotations which are target for alignment, suppose w2Ri_ in world-2 frame for all frames i.

    Returns:
        transformed rotations which have the same origin as reference (now living in world-2 frame)
    """
    w1Ri0 = input_list[0]
    i0Rw1 = w1Ri0.inverse()
    w2Ri0 = ref_list[0]
    # origin_transform -- map the origin of the input list to the reference list
    w2Rw1 = w2Ri0.compose(i0Rw1)

    # apply the coordinate shift to all entries in input
    return [w2Rw1.compose(w1Ri) for w1Ri in input_list]


def align_translations(input_list: List[Point3], ref_list: List[Point3]) -> Tuple[List[Point3], float, Rot3, Point3]:
    """Aligns the list of translations to the reference list applying scale, rotation, and shift.

    The motion model to be solved is w2ti = scale @ R @ w1ti + t.

    References:
        1. Umeyama, Shinji. "Least-squares estimation of transformation parameters between two point patterns." IEEE
           Computer Architecture Letters 13.04 (1991): 376-380.
        2. Zhang, Zichao, and Davide Scaramuzza. "A tutorial on quantitative trajectory evaluation for visual
           (-inertial) odometry." 2018 IEEE RSJ International Conference on Intelligent Robots and Systems (IROS).
           IEEE, 2018.


    Args:
        input_list: input point3s which need to be aligned, suppose w1ti in world-1 frame for all frames i.
        ref_list: reference point3s which are target for alignment, suppose w2ti_ in world-2 frame for all frames i.

    Returns:
        transformed point3s which has minimum error from the reference (now living in world-2 frame).
        Scale factor of the transformation.
        Rotation of the transformation.
        Origin shift (translation) of the transformation.
    """
    # TODO: use GTSAM's align function when it has been merged in develop.
    N = len(input_list)

    # convert input to 2D arrays, with rows as individual poses
    input_pose_matrix = np.vstack([x.reshape(1, 3) for x in input_list])
    ref_pose_matrix = np.vstack([x.reshape(1, 3) for x in ref_list])

    # compute the mean pose
    input_mean = np.mean(input_pose_matrix, axis=0)
    ref_mean = np.mean(ref_pose_matrix, axis=0)

    # subtract the means
    input_pose_matrix = input_pose_matrix - input_mean.reshape(1, 3)
    ref_pose_matrix = ref_pose_matrix - ref_mean.reshape(1, 3)

    # compute the variances
    input_variance = np.sum(np.square(input_pose_matrix), axis=None) / N
    # ref_variance = np.sum(np.square(ref_pose_matrix), axis=None) / N

    # compute the covariance matrix and perform SVD
    cov_mat = np.dot(ref_pose_matrix.T, input_pose_matrix) / N
    U, S, Vt = np.linalg.svd(cov_mat)
    S = np.diag(S)

    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1

    rot_matrix = U @ W @ Vt
    scale = np.trace(S @ W) / input_variance
    origin_translation = ref_mean - scale * rot_matrix @ input_mean

    return (
        [scale * (rot_matrix @ x) + origin_translation for x in input_list],
        scale,
        Rot3(rot_matrix),
        origin_translation,
    )


def align_poses(input_list: List[Pose3], ref_list: List[Pose3]) -> List[Pose3]:
    """Aligns the input list of poses to the reference list by aligning rotations followed by translations. The two
    alignments are done independently.

    Args:
        input_list: input pose3s which need to be aligned, suppose w1Ti in world-1 frame for all frames i.
        ref_list: reference pose3s which are target for alignment, suppose w2Ti_ in world-2 frame for all frames i.

    Returns:
         transformed pose3s which has minimum euclidean error from the reference (now living in world-2 frame).
    """
    # TODO: this doesn't make sense and we need to build a joint optimization.

    # align rotations first
    input_rotations = [x.rotation() for x in input_list]
    ref_rotations = [x.rotation() for x in ref_list]
    aligned_rotations = align_rotations(input_rotations, ref_rotations)

    # apply the rotations to the whole pose (i.e translation too)
    rel_rotations = [x.between(y) for x, y in zip(input_rotations, aligned_rotations)]
    intermediate_list = [x.compose(Pose3(y, np.zeros(3))) for x, y in zip(input_list, rel_rotations)]

    # align the translations
    input_translations = [x.translation() for x in intermediate_list]
    ref_translations = [x.translation() for x in ref_list]
    aligned_translations, _, _, _ = align_translations(input_translations, ref_translations)

    return [Pose3(x, y) for x, y in zip(aligned_rotations, aligned_translations)]


def compare_rotations(input_list: List[Optional[Rot3]], ref_list: List[Optional[Rot3]], angle_threshold: float) -> bool:
    """Helper function to compare two lists of Rot3s using angle of relative rotation at each index.

    Notes:
    1. The input lists have the rotations in the same order, and can contain None entries.

    Args:
        input_list: 1st list of rotations.
        ref_list: 2nd list of rotations.
        angle_threshold: threshold of relative rotation angle for equality.

    Returns:
        Result of the comparison.
    """

    if len(input_list) != len(ref_list):
        return False

    for R, R_ in zip(input_list, ref_list):
        if R is not None and R_ is not None:
            if compute_relative_rotation_angle(R, R_) > angle_threshold:
                return False
        elif R is not None or R_ is not None:
            return False

    return True


def align_and_compare_rotations(
    input_list: List[Optional[Rot3]], ref_list: List[Optional[Rot3]], angle_threshold: float
) -> bool:
    """Wrapper combining align_rotations and compare_rotations.

    Notes:
    1. The input lists have the rotations in the same order, and can contain None entries.

    Args:
        input_list: 1st list of rotations.
        ref_list: 2nd list of rotations.
        angle_threshold: threshold of relative rotation angle for equality.

    Returns:
        result of the comparison post alignment.
    """

    if len(input_list) != len(ref_list):
        return False

    # remove none entries and align
    valid_input_list = []
    valid_ref_list = []
    for R, R_ in zip(input_list, ref_list):
        if R is not None and R_ is not None:
            valid_input_list.append(R)
            valid_ref_list.append(R_)
        elif R is not None or R_ is not None:
            return False
    aligned_input_list = align_rotations(valid_input_list, valid_ref_list)

    # finally, compare
    return compare_rotations(aligned_input_list, valid_ref_list, angle_threshold)


def compare_translations(
    input_list: List[Optional[Point3]],
    ref_list: List[Optional[Point3]],
    relative_error_thresh: float = 4e-1,
    absolute_error_thresh: float = 1e-1,
) -> bool:
    """Helper function to compare two lists of Point3s using L2 distances at each index.

    Notes:
    1. The input lists have the translations in the same order, and can contain None entries.

    Args:
        input_list: 1st list of translations.
        ref_list: 2nd list of translations.
        relative_error_thresh (optional): relative error threshold for comparisons. Defaults to 4e-1.
        absolute_error_thresh (optional): absolute error threshold for comparisons. Defaults to 1e-1.

    Returns:
        Results of the comparison.
    """

    # check the length of the input lists
    if len(input_list) != len(ref_list):
        return False

    for t, t_ in zip(input_list, ref_list):
        if t is not None and t_ is not None:
            if not np.allclose(t, t_, rtol=relative_error_thresh, atol=absolute_error_thresh):
                return False
        elif t is not None or t_ is not None:
            return False

    return True


def align_and_compare_translations(
    input_list: List[Optional[Point3]],
    ref_list: List[Optional[Point3]],
    relative_error_thresh: float = 4e-1,
    absolute_error_thresh: float = 1e-1,
) -> bool:
    """Wrapper combining align_translations and compare_translations.

    Notes:
    1. The input lists have the translations in the same order, and can contain None entries.

    Args:
        input_list: 1st list of rotations.
        ref_list: 2nd list of rotations.
        relative_error_thresh (optional): relative error threshold for comparisons. Defaults to 4e-1.
        absolute_error_thresh (optional): absolute error threshold for comparisons. Defaults to 1e-1.

    Returns:
        result of the comparison post alignment.
    """

    if len(input_list) != len(ref_list):
        return False

    # remove none entries and align
    valid_input_list = []
    valid_ref_list = []
    for t, t_ in zip(input_list, ref_list):
        if t is not None and t_ is not None:
            valid_input_list.append(t)
            valid_ref_list.append(t_)
        elif t is not None or t_ is not None:
            return False
    aligned_input_list, _, _, _ = align_translations(valid_input_list, valid_ref_list)

    # finally, compare
    return compare_translations(aligned_input_list, valid_ref_list, relative_error_thresh, absolute_error_thresh)


def compute_relative_rotation_angle(R_1: Optional[Rot3], R_2: Optional[Rot3]) -> Optional[float]:
    """Compute the angle between two rotations.

    Note: the angle is the norm of the angle-axis representation.

    Args:
        R_1: the first rotation.
        R_2: the second rotation.

    Returns:
        the angle between two rotations, in degrees
    """

    if R_1 is None or R_2 is None:
        return None

    relative_rot = R_1.between(R_2)
    relative_rot_angle_rad = relative_rot.axisAngle()[1]
    relative_rot_angle_deg = np.rad2deg(relative_rot_angle_rad)
    return relative_rot_angle_deg


def compute_relative_unit_translation_angle(U_1: Optional[Unit3], U_2: Optional[Unit3]) -> Optional[float]:
    """Compute the angle between two unit-translations.

    Args:
        U_1: the first unit-translation.
        U_2: the second unit-translation.

    Returns:
        the angle between the two unit-vectors, in degrees
    """
    if U_1 is None or U_2 is None:
        return None

    # TODO: expose Unit3's dot function and use it directly
    dot_product = np.dot(U_1.point3(), U_2.point3())
    dot_product = np.clip(dot_product, -1, 1)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg


def compute_translation_to_direction_angle(
    i2Ui1: Optional[Unit3], wTi2: Optional[Pose3], wTi1: Optional[Pose3]
) -> Optional[float]:
    """Compute angle between a unit translation and the relative translation between 2 poses.

    Given a unit translation measurement from i2 to i1, the estimated poses of
    i1 and i2, returns the angle between the relative position of i1 wrt i2
    and the unit translation measurement.

    Args:
        i2Ui1: Unit translation measurement.
        wTi2: Pose of camera i2.
        wTi1: Pose of camera i1.

    Returns:
        Angle between measurement and relative estimated translation in degrees.
    """
    if i2Ui1 is None or wTi2 is None or wTi1 is None:
        return None

    i2Ti1 = wTi2.between(wTi1)
    i2Ui1_estimated = Unit3(i2Ti1.translation())
    return compute_relative_unit_translation_angle(i2Ui1, i2Ui1_estimated)


def compute_points_distance_l2(wti1: Optional[Point3], wti2: Optional[Point3]) -> Optional[float]:
    """Computes the L2 distance between the two input 3D points.

    Assumes the points are in the same coordinate frame. Returns None if either
    point is None.

    Args:
        wti1: Point1 in world frame
        wti2: Point2 in world frame

    Returns:
        L2 norm of wti1 - wti2
    """
    if wti1 is None or wti2 is None:
        return None
    return np.linalg.norm(wti1 - wti2)
