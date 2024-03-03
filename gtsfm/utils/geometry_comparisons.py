"""Utility functions for comparing different types related to geometry.

Authors: Ayush Baid, John Lambert
"""

from typing import List, Optional

import numpy as np
from gtsam import Point3, Pose3, Rot3, Unit3
from scipy.spatial.transform import Rotation

import gtsfm.utils.logger as logger_utils
import gtsfm.utils.alignment as alignment_utils

EPSILON = np.finfo(float).eps

logger = logger_utils.get_logger()

MAX_ALIGNMENT_ERROR = np.finfo(np.float32).max


def compare_rotations(
    aRi_list: List[Optional[Rot3]], bRi_list: List[Optional[Rot3]], angular_error_threshold_degrees: float
) -> bool:
    """Helper function to compare two lists of global Rot3, after aligning them.

    Notes:
    1. The input lists have the rotations in the same order, and can contain None entries.

    Args:
        aTi_list: 1st list of rotations.
        bTi_list: 2nd list of rotations.
        angular_error_threshold_degrees: the threshold for angular error between two rotations.
    Returns:
        Result of the comparison.
    """
    if len(aRi_list) != len(bRi_list):
        return False

    # check the presense of valid Rot3 objects in the same location
    aRi_valid = [i for (i, aRi) in enumerate(aRi_list) if aRi is not None]
    bRi_valid = [i for (i, bRi) in enumerate(bRi_list) if bRi is not None]
    if aRi_valid != bRi_valid:
        return False

    if len(aRi_valid) <= 1:
        # we need >= two entries going forward for meaningful comparisons
        return False

    aRi_list = [aRi_list[i] for i in aRi_valid]
    bRi_list = [bRi_list[i] for i in bRi_valid]

    # frame 'a' is the target/reference, and bRi_list will be transformed
    aRi_list_ = alignment_utils.align_rotations(aRi_list, bRi_list)
    relative_rotations_angles = np.array(
        [compute_relative_rotation_angle(aRi, aRi_) for (aRi, aRi_) in zip(aRi_list, aRi_list_)], dtype=np.float32
    )
    return np.all(relative_rotations_angles < angular_error_threshold_degrees)


def compare_global_poses(
    aTi_list: List[Optional[Pose3]],
    bTi_list: List[Optional[Pose3]],
    rot_angular_error_thresh_degrees: float = 2,
    trans_err_atol: float = 1e-2,
    trans_err_rtol: float = 1e-1,
    verbose: bool = True,
) -> bool:
    """Helper function to compare two lists of Point3s using L2 distances at each index.

    Notes:
    1. The input lists have the poses in the same order, and can contain None entries.
    2. To resolve global origin ambiguity, we fit a Sim(3) transformation and align the two pose graphs.

    Args:
        aTi_list: 1st list of poses.
        bTi_list: 2nd list of poses.
        rot_angular_error_threshold_degrees (optional): angular error threshold for rotations. Defaults to 2.
        trans_err_atol (optional): Absolute error threshold for translation. Defaults to 1e-2.
        trans_err_rtol (optional): Relative error threshold for translation. Defaults to 1e-1.

    Returns:
        Result of the comparison.
    """

    # Check the length of the input lists
    if len(aTi_list) != len(bTi_list):
        return False

    # Check the presence of valid Pose3 objects in the same location.
    aTi_valid = [i for (i, aTi) in enumerate(aTi_list) if aTi is not None]
    bTi_valid = [i for (i, bTi) in enumerate(bTi_list) if bTi is not None]
    if aTi_valid != bTi_valid:
        return False

    if len(aTi_valid) <= 1:
        # we need >= two entries going forward for meaningful comparisons
        return False

    # Align the remaining poses.
    aTi_list = [aTi_list[i] for i in aTi_valid]
    bTi_list = [bTi_list[i] for i in bTi_valid]

    #  We set frame "a" the target/reference
    aTi_list_, _ = alignment_utils.align_poses_sim3_exhaustive(aTi_list, bTi_list)

    rotations_equal = all(
        [
            compute_relative_rotation_angle(aTi.rotation(), aTi_.rotation()) < rot_angular_error_thresh_degrees
            for (aTi, aTi_) in zip(aTi_list, aTi_list_)
        ]
    )
    translations_equal = all(
        [
            np.allclose(aTi.translation(), aTi_.translation(), atol=trans_err_atol, rtol=trans_err_rtol)
            for (aTi, aTi_) in zip(aTi_list, aTi_list_)
        ]
    )
    if verbose:
        rotation_errors = np.array(
            [
                compute_relative_rotation_angle(aTi.rotation(), aTi_.rotation())
                for (aTi, aTi_) in zip(aTi_list, aTi_list_)
            ]
        )
        translation_errors = np.array(
            [np.linalg.norm(aTi.translation() - aTi_.translation()) for (aTi, aTi_) in zip(aTi_list, aTi_list_)]
        )
        logger.info("Comparison Rotation Errors (degrees): " + str(np.round(rotation_errors, 2)))
        logger.info("Comparison Translation Errors: " + str(np.round(translation_errors, 2)))

    return rotations_equal and translations_equal


def compute_relative_rotation_angle(R_1: Optional[Rot3], R_2: Optional[Rot3]) -> Optional[float]:
    """Compute the angle between two rotations.

    Note: the angle is the norm of the angle-axis representation.

    Args:
        R_1: The first rotation.
        R_2: The second rotation.

    Returns:
        The angle between two rotations, in degrees.
    """

    if R_1 is None or R_2 is None:
        return None

    relative_rot = R_1.between(R_2)
    # TODO(johnwlambert): we are using scipy.spatial.transform as a hotfix until GTSAM axisAngle() is patched.
    # See https://github.com/borglab/gtsam/issues/886
    scaled_axis = Rotation.from_matrix(relative_rot.matrix()).as_rotvec()
    relative_rot_angle_rad = np.linalg.norm(scaled_axis)
    relative_rot_angle_deg = np.rad2deg(relative_rot_angle_rad)
    return relative_rot_angle_deg


def compute_relative_unit_translation_angle(U_1: Optional[Unit3], U_2: Optional[Unit3]) -> Optional[float]:
    """Compute the angle between two unit-translations.

    Args:
        U_1: The first unit-translation.
        U_2: The second unit-translation.

    Returns:
        The angle between the two unit-vectors, in degrees.
    """
    if U_1 is None or U_2 is None:
        return None

    # TODO: expose Unit3's dot function and use it directly.
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
        wti1: Point1 in world frame.
        wti2: Point2 in world frame.

    Returns:
        L2 norm of wti1 - wti2
    """
    if wti1 is None or wti2 is None:
        return None
    return np.linalg.norm(wti1 - wti2)


def compute_cyclic_rotation_error(i1Ri0: Rot3, i2Ri1: Rot3, i2Ri0: Rot3) -> float:
    """Computes the cycle error in degrees after composing the three input rotations.

    The cyclic error is the angle between identity and the rotation obtained by composing the three input relative
    rotations, i.e., (I - inv(i2Ri0) * i2Ri1 * i1Ri0).

    Args:
        i1Ri0: Relative rotation of camera i0 with respect to i1.
        i2Ri1: Relative rotation of camera i1 with respect to i2.
        i2Ri0: Relative rotation of camera i0 with respect to i2.

    Returns:
        Cyclic rotation error in degrees.
    """
    i0Ri0_from_cycle = i2Ri0.inverse().compose(i2Ri1).compose(i1Ri0)
    return compute_relative_rotation_angle(Rot3(), i0Ri0_from_cycle)


def get_points_within_radius_of_cameras(
    wTi_list: List[Pose3], points_3d: np.ndarray, radius: float = 50
) -> Optional[np.ndarray]:
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
    for j, wTi in enumerate(wTi_list):
        is_nearby_matrix[:, j] = np.linalg.norm(points_3d - wTi.translation(), axis=1) < radius

    is_nearby_to_any_cam = np.any(is_nearby_matrix, axis=1)
    nearby_points_3d = points_3d[is_nearby_to_any_cam]
    return nearby_points_3d


def is_valid_SO3(R: Rot3) -> bool:
    """Verifies that provided rotation matrix is a valid member of SO(3)."""
    R = R.matrix()
    is_unit_det = np.isclose(np.linalg.det(R), 1.0)
    is_orthogonal = np.allclose(R @ R.T, np.eye(3))
    return is_unit_det and is_orthogonal
