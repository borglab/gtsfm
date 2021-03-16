"""Utility functions for comparing different types related to geometry.

Authors: Ayush Baid
"""
from typing import List, Optional

import numpy as np
from gtsam import Point3, Pose3, Pose3Pairs, Rot3, Similarity3, Unit3
from scipy.spatial.transform import Rotation

from gtsfm.utils.logger import get_logger

EPSILON = np.finfo(float).eps

logger = get_logger()

def align_rotations(aRi_list: List[Rot3], bRi_list: List[Rot3]) -> List[Rot3]:
    """Aligns the list of rotations to the reference list by shifting origin.

    TODO (John): replace later with Karcher mean to account for noisy estimates

    Args:
        aRi_list: reference rotations in frame "a" which are the targets for alignment
        bRi_list: input rotations which need to be aligned to frame "a"
            
    Returns:
        aRi_list_: transformed input rotations previously "bRi_list" but now which
            have the same origin as reference (now living in "a" frame)
    """
    aRi0 = aRi_list[0]

    bRi0 = bRi_list[0]
    i0Rb = bRi0.inverse()

    # origin_transform -- map the origin of the input list to the reference list
    aRb = aRi0.compose(i0Rb)

    # apply the coordinate shift to all entries in input
    return [aRb.compose(bRi) for bRi in bRi_list]


def align_poses_sim3(aTi_list: List[Pose3], bTi_list: List[Pose3]) -> List[Pose3]:
    """Align by similarity transformation.

    We force SIM(3) alignment rather than SE(3) alignment.
    We assume the two trajectories are of the exact same length.

    Args:
        aTi_list: reference poses in frame "a" which are the targets for alignment
        bTi_list: input poses which need to be aligned to frame "a"
            
    Returns:
        aTi_list_: transformed input poses previously "bTi_list" but now which
            have the same origin and scale as reference (now living in "a" frame)
    """
    n_to_align = len(aTi_list)
    assert len(aTi_list) == len(bTi_list)
    assert n_to_align >= 2, "SIM(3) alignment uses at least 2 frames"

    ab_pairs = Pose3Pairs(list(zip(aTi_list, bTi_list)))

    aSb = Similarity3.Align(ab_pairs)

    # provide a summary of the estimated alignment transform
    aRb = aSb.rotation().matrix()
    atb = aSb.translation()
    rz,ry,rx = Rotation.from_matrix(aRb).as_euler('zyx', degrees=True)
    logger.info(f"Sim(3) Rotation `aRb`: rz={rz:.2f} deg., ry={ry:.2f} deg., rx={rx:.2f} deg.", )
    logger.info(f"Sim(3) Translation `atb`: [tx,ty,tz]={str(np.round(atb,2))}")
    logger.info(f"Sim(3) Scale `asb`: {float(aSb.scale()):.2f}")

    aTi_list_ = []
    for i in range(n_to_align):
        bTi = bTi_list[i]

        aTi_list_.append(aSb.transformFrom(bTi))

    logger.info("Pose graph Sim(3) alignment complete.")

    return aTi_list_


def compare_rotations(aRi_list: List[Optional[Rot3]], bRi_list_: List[Optional[Rot3]]) -> bool:
    """Helper function to compare two lists of global Rot3, considering the
    origin as ambiguous.

    Notes:
    1. The input lists have the rotations in the same order, and can contain None entries.
    2. To resolve global origin ambiguity, we will fix one image index as origin in both the inputs and transform both
       the lists to the new origins.

    Args:
        aTi_list: 1st list of rotations.
        bTi_list: 2nd list of rotations.
    Returns:
        result of the comparison.
    """
    if len(aRi_list) != len(bRi_list_):
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
    aRi_list_ = align_rotations(aRi_list, bRi_list)

    return all([aRi.equals(aRi_, 1e-1) for (aRi, aRi_) in zip(aRi_list, aRi_list_)])


def compare_global_poses(
    aTi_list: List[Optional[Pose3]],
    bTi_list: List[Optional[Pose3]],
    rot_err_thresh: float = 1e-3,
    trans_err_thresh: float = 1e-1,
) -> bool:
    """Helper function to compare two lists of global Pose3, considering the
    origin and scale ambiguous.

    Notes:
    1. The input lists have the poses in the same order, and can contain None entries.
    2. To resolve global origin ambiguity, we fit a Sim(3) transformation and align the two pose graphs

    Args:
        aTi_list: 1st list of poses.
        bTi_list: 2nd list of poses.
        rot_err_thresh (optional): error threshold for rotations. Defaults to 1e-3.
        trans_err_thresh (optional): relative error threshold for translation. Defaults to 1e-1.

    Returns:
        result of the comparison.
    """

    # check the length of the input lists
    if len(aTi_list) != len(bTi_list):
        return False

    # check the presense of valid Pose3 objects in the same location
    aTi_valid = [i for (i, aTi) in enumerate(aTi_list) if aTi is not None]
    bTi_valid = [i for (i, bTi) in enumerate(bTi_list) if bTi is not None]
    if aTi_valid != bTi_valid:
        return False

    if len(aTi_valid) <= 1:
        # we need >= two entries going forward for meaningful comparisons
        return False

    # align the remaining poses
    aTi_list = [aTi_list[i] for i in aTi_valid]
    bTi_list = [bTi_list[i] for i in bTi_valid]

    #  We set frame "a" the target/reference
    aTi_list_ = align_poses_sim3(aTi_list, bTi_list)

    return all(
        [
            (
                aTi.rotation().equals(aTi_.rotation(), rot_err_thresh)
                and np.allclose(
                    aTi.translation(),
                    aTi_.translation(),
                    rtol=trans_err_thresh,
                )
            )
            for (aTi, aTi_) in zip(aTi_list, aTi_list_)
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


