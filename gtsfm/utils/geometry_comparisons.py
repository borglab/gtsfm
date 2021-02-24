"""Utility functions for comparing different types related to geometry.

Authors: Ayush Baid
"""
from typing import List, Optional

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


def align_poses(input_list: List[Pose3], ref_list: List[Pose3]) -> List[Pose3]:
    """Aligns the list of poses to the reference list by shifting origin and
    scaling translations.

    Args:
        input_list: input poses which need to be aligned, suppose w1Ti in world-1 frame for all frames i.
        ref_list: reference poses which are target for alignment, suppose w2Ti_ in world-2 frame for all frames i.

    Returns:
        transformed poses which have the same origin and scale as reference (now living in world-2 frame)
    """
    # match the scales first
    wTi0 = input_list[0]
    input_distances = np.array([np.linalg.norm((wTi.between(wTi0)).translation()) for wTi in input_list[1:]])

    wTi0 = ref_list[0]
    ref_distances = np.array([np.linalg.norm((wTi.between(wTi0)).translation()) for wTi in ref_list[1:]]) + EPSILON

    # rescale poses to account for SfM scale ambiguity
    scales = ref_distances / input_distances
    scaling_factor = np.median(scales)

    scaled_list = [Pose3(w2Ti.rotation(), w2Ti.translation() * scaling_factor) for w2Ti in input_list]

    # now match origin
    w1Ti0 = scaled_list[0]
    i0Tw1 = w1Ti0.inverse()
    w2Ti0_ = ref_list[0]
    # origin transform -- map the origin of the input list to the reference list
    w2Tw1 = w2Ti0_.compose(i0Tw1)

    scaled_shifted_list = [w2Tw1.compose(w1Ti) for w1Ti in scaled_list]

    return scaled_shifted_list


def compare_rotations(wRi_list: List[Optional[Rot3]], wRi_list_: List[Optional[Rot3]]) -> bool:
    """Helper function to compare two lists of global Rot3, considering the
    origin as ambiguous.

    Notes:
    1. The input lists have the rotations in the same order, and can contain None entries.
    2. To resolve global origin ambiguity, we will fix one image index as origin in both the inputs and transform both
       the lists to the new origins.

    Args:
        wRi_list: 1st list of rotations.
        wRi_list_: 2nd list of rotations.

    Returns:
        result of the comparison.
    """

    if len(wRi_list) != len(wRi_list_):
        return False

    # check the presense of valid Rot3 objects in the same location
    wRi_valid = [i for (i, wRi) in enumerate(wRi_list) if wRi is not None]
    wRi_valid_ = [i for (i, wRi) in enumerate(wRi_list_) if wRi is not None]
    if wRi_valid != wRi_valid_:
        return False

    if len(wRi_valid) <= 1:
        # we need >= two entries going forward for meaningful comparisons
        return False

    wRi_list = [wRi_list[i] for i in wRi_valid]
    wRi_list_ = [wRi_list_[i] for i in wRi_valid_]

    wRi_list = align_rotations(wRi_list, ref_list=wRi_list_)

    return all([wRi.equals(wRi_, 1e-1) for (wRi, wRi_) in zip(wRi_list, wRi_list_)])


def compare_global_poses(
    wTi_list: List[Optional[Pose3]],
    wTi_list_: List[Optional[Pose3]],
    rot_err_thresh: float = 1e-3,
    trans_err_thresh: float = 1e-1,
) -> bool:
    """Helper function to compare two lists of global Pose3, considering the
    origin and scale ambiguous.

    Notes:
    1. The input lists have the poses in the same order, and can contain None entries.
    2. To resolve global origin ambiguity, we will fix one image index as origin in both the inputs and transform both
       the lists to the new origins.
    3. As there is a scale ambiguity, we use the median scaling factor to resolve the ambiguity.

    Args:
        wTi_list: 1st list of poses.
        wTi_list_: 2nd list of poses.
        rot_err_thresh (optional): error threshold for rotations. Defaults to 1e-3.
        trans_err_thresh (optional): relative error threshold for translation. Defaults to 1e-1.

    Returns:
        results of the comparison.
    """

    # check the length of the input lists
    if len(wTi_list) != len(wTi_list_):
        return False

    # check the presense of valid Pose3 objects in the same location
    wTi_valid = [i for (i, wTi) in enumerate(wTi_list) if wTi is not None]
    wTi_valid_ = [i for (i, wTi) in enumerate(wTi_list_) if wTi is not None]
    if wTi_valid != wTi_valid_:
        return False

    if len(wTi_valid) <= 1:
        # we need >= two entries going forward for meaningful comparisons
        return False

    # align the remaining poses
    wTi_list = [wTi_list[i] for i in wTi_valid]
    wTi_list_ = [wTi_list_[i] for i in wTi_valid_]

    wTi_list = align_poses(wTi_list, ref_list=wTi_list_)

    return all(
        [
            (
                wTi.rotation().equals(wTi_.rotation(), rot_err_thresh)
                and np.allclose(
                    wTi.translation(),
                    wTi_.translation(),
                    rtol=trans_err_thresh,
                )
            )
            for (wTi, wTi_) in zip(wTi_list, wTi_list_)
        ]
    )


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


def get_points_within_radius_of_cameras(wTi_list: List[Pose3], points_3d: np.ndarray, radius: float = 50) -> Optional[np.ndarray]:
    """Return those 3d points that fall within a specified radius from any camera.
    Args:
        wTi_list: camera poses
        points_3d: array of shape (N,3) representing 3d points
        radius: distance threshold, in meters

    Returns:
        nearby_points_3d: array of shape (M,3), where M <= N
    """
    if len(wTi_list) == 0 or points_3d.size == 0 or radius <= 0:
        return None

    num_points = points_3d.shape[0]
    num_poses = len(wTi_list)
    # each row represents attributes for a single point
    # each column represents
    is_nearby_matrix = np.zeros((num_points, num_poses), dtype=bool)
    for j,wTi in enumerate(wTi_list):
        is_nearby_matrix[:,j] = np.linalg.norm(points_3d - wTi.translation(), axis=1) < radius

    is_nearby_to_any_cam = np.any(is_nearby_matrix, axis=1)
    nearby_points_3d = points_3d[is_nearby_to_any_cam]
    return nearby_points_3d


