"""Utility functions for aligning different geometry types

Authors: Ayush Baid, John Lambert
"""

import copy
from typing import List, Optional, Tuple

import gtsam
import numpy as np
from gtsam import Pose3, Pose3Pairs, Rot3, Rot3Vector, Similarity3
from scipy.spatial.transform import Rotation

import gtsfm.utils.metrics as metric_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData

EPSILON = np.finfo(float).eps

logger = logger_utils.get_logger()

MAX_ALIGNMENT_ERROR = np.finfo(np.float32).max
MAX_NUM_HYPOTHESIS_FOR_ROBUST_ALIGNMENT: int = 200


def align_rotations(aRi_list: List[Optional[Rot3]], bRi_list: List[Optional[Rot3]]) -> List[Rot3]:
    """Aligns the list of rotations to the reference list by using Karcher mean.

    Args:
        aRi_list: Reference rotations in frame "a" which are the targets for alignment
        bRi_list: Input rotations which need to be aligned to frame "a"

    Returns:
        aRi_list_: Transformed input rotations previously "bRi_list" but now which
            have the same origin as reference (now living in "a" frame)
    """
    aRb_list = [
        aRi.compose(bRi.inverse()) for aRi, bRi in zip(aRi_list, bRi_list) if aRi is not None and bRi is not None
    ]
    if len(aRb_list) > 0:
        aRb = gtsam.FindKarcherMean(Rot3Vector(aRb_list))
    else:
        aRb = Rot3()

    # Apply the coordinate shift to all entries in input.
    return [aRb.compose(bRi) if bRi is not None else None for bRi in bRi_list]


def align_poses_sim3_ignore_missing(
    aTi_list: List[Optional[Pose3]], bTi_list: List[Optional[Pose3]]
) -> Tuple[List[Optional[Pose3]], Similarity3]:
    """Align by similarity transformation, but allow missing estimated poses in the input.

    Note: this is a wrapper for align_poses_sim3() that allows for missing poses/dropped cameras.
    This is necessary, as align_poses_sim3() requires a valid pose for every input pair.

    We force SIM(3) alignment rather than SE(3) alignment.
    We assume the two trajectories are of the exact same length.

    Args:
        aTi_list: reference poses in frame "a" which are the targets for alignment
        bTi_list: input poses which need to be aligned to frame "a"

    Returns:
        aTi_list_: transformed input poses previously "bTi_list" but now which
            have the same origin and scale as reference (now living in "a" frame)
        aSb: Similarity(3) object that aligns the two pose graphs.
    """
    assert len(aTi_list) == len(bTi_list)

    # only choose target poses for which there is a corresponding estimated pose
    corresponding_aTi_list = []
    valid_camera_idxs = []
    valid_bTi_list = []
    for i, bTi in enumerate(bTi_list):
        if bTi is not None:
            valid_camera_idxs.append(i)
            valid_bTi_list.append(bTi)
            corresponding_aTi_list.append(aTi_list[i])

    valid_aTi_list_, aSb = align_poses_sim3_robust(aTi_list=corresponding_aTi_list, bTi_list=valid_bTi_list)

    num_cameras = len(aTi_list)
    # now at valid indices
    aTi_list_ = [None] * num_cameras
    for i in range(num_cameras):
        if i in valid_camera_idxs:
            aTi_list_[i] = valid_aTi_list_.pop(0)

    return aTi_list_, aSb


def align_poses_sim3_exhaustive(aTi_list: List[Pose3], bTi_list: List[Pose3]) -> Tuple[List[Pose3], Similarity3]:
    """Align two pose graphs via similarity transformation by trying out all pairs for alignment and picking the best
    fit.
    Note: poses cannot be missing/invalid.

    We force Sim(3) alignment rather than SE(3) alignment.
    We assume the two trajectories are of the exact same length.

    Args:
        aTi_list: reference poses in frame "a" which are the targets for alignment
        bTi_list: input poses which need to be aligned to frame "a"

    Returns:
        aTi_list_: transformed input poses previously "bTi_list" but now which
            have the same origin and scale as reference (now living in "a" frame)
        aSb: Similarity(3) object that aligns the two pose graphs.
    """
    assert len(aTi_list) == len(bTi_list)
    n_to_align = len(aTi_list)
    if n_to_align < 2:
        logger.error("SIM(3) alignment uses at least 2 frames; Skipping")
        return bTi_list, Similarity3(Rot3(), np.zeros((3,)), 1.0)

    best_pose_auc_5deg = 0
    best_aSb = Similarity3()
    for i in range(n_to_align):
        for j in range(i + 1, n_to_align):
            aTi_sample = copy.deepcopy([aTi_list[i], aTi_list[j]])
            bTi_sample = copy.deepcopy([bTi_list[i], bTi_list[j]])

            _, aSb_candidate = align_poses_sim3(aTi_sample, bTi_sample)

            aTi_candidate_: List[Pose3] = [aSb_candidate.transformFrom(bTi) for bTi in bTi_list]

            candidate_metrics = metric_utils.compute_ba_pose_metrics(
                gt_wTi_list=aTi_list,
                computed_wTi_list=aTi_candidate_,
            )
            pose_auc_5deg = candidate_metrics.metrics[6].data.item()

            if pose_auc_5deg > best_pose_auc_5deg:
                logger.debug("Update auc: %.2f -> %.2f", best_pose_auc_5deg, pose_auc_5deg)
                best_pose_auc_5deg = pose_auc_5deg
                best_aSb = aSb_candidate

                aRb = best_aSb.rotation().matrix()
                atb = best_aSb.translation()
                rz, ry, rx = Rotation.from_matrix(aRb).as_euler("zyx", degrees=True)
                logger.debug("Sim(3) Rotation `aRb`: rz=%.2f deg., ry=%.2f deg., rx=%.2f deg.", rz, ry, rx)
                logger.debug("Sim(3) Translation `atb`: [%.2f, %.2f, %.2f]", atb[0], atb[1], atb[2])
                logger.debug("Sim(3) Scale `asb`: %.2f", float(best_aSb.scale()))

    aRb = best_aSb.rotation().matrix()
    atb = best_aSb.translation()
    rz, ry, rx = Rotation.from_matrix(aRb).as_euler("zyx", degrees=True)
    logger.info("Sim(3) Rotation `aRb`: rz=%.2f deg., ry=%.2f deg., rx=%.2f deg.", rz, ry, rx)
    logger.info("Sim(3) Translation `atb`: [%.2f, %.2f, %.2f]", atb[0], atb[1], atb[2])
    logger.info("Sim(3) Scale `asb`: %.2f", float(best_aSb.scale()))

    best_aTi_: List[Pose3] = [best_aSb.transformFrom(bTi) for bTi in bTi_list]
    return best_aTi_, best_aSb


def align_poses_sim3_robust(aTi_list: List[Pose3], bTi_list: List[Pose3]) -> Tuple[List[Pose3], Similarity3]:
    """Align two pose graphs via similarity transformation by trying out some samples of pairs of cameras for candidate
    transforms.
    Note: poses cannot be missing/invalid.

    We force Sim(3) alignment rather than SE(3) alignment.
    We assume the two trajectories are of the exact same length.

    Args:
        aTi_list: reference poses in frame "a" which are the targets for alignment
        bTi_list: input poses which need to be aligned to frame "a"

    Returns:
        aTi_list_: transformed input poses previously "bTi_list" but now which
            have the same origin and scale as reference (now living in "a" frame)
        aSb: Similarity(3) object that aligns the two pose graphs.
    """
    assert len(aTi_list) == len(bTi_list)
    n_to_align = len(aTi_list)
    if n_to_align < 2:
        logger.error("SIM(3) alignment uses at least 2 frames; Skipping")
        return bTi_list, Similarity3(Rot3(), np.zeros((3,)), 1.0)

    max_possible_hypothesis: int = (n_to_align * (n_to_align - 1)) // 2
    if max_possible_hypothesis <= MAX_NUM_HYPOTHESIS_FOR_ROBUST_ALIGNMENT:
        return align_poses_sim3_exhaustive(aTi_list=aTi_list, bTi_list=bTi_list)

    best_pose_auc_5deg = 0
    best_aSb = Similarity3()

    sample_pose_pairs = np.random.choice(
        len(aTi_list),
        size=MAX_NUM_HYPOTHESIS_FOR_ROBUST_ALIGNMENT * 2,
        replace=True,
    ).reshape(-1, 2)

    for i, j in sample_pose_pairs:
        aTi_sample = copy.deepcopy([aTi_list[i], aTi_list[j]])
        bTi_sample = copy.deepcopy([bTi_list[i], bTi_list[j]])

        _, aSb_candidate = align_poses_sim3(aTi_sample, bTi_sample)

        aTi_candidate_: List[Pose3] = [aSb_candidate.transformFrom(bTi) for bTi in bTi_list]

        candidate_metrics = metric_utils.compute_ba_pose_metrics(
            gt_wTi_list=aTi_list,
            computed_wTi_list=aTi_candidate_,
        )
        pose_auc_5deg = candidate_metrics.metrics[6].data.item()

        if pose_auc_5deg > best_pose_auc_5deg:
            logger.debug("Update auc: %.2f -> %.2f", best_pose_auc_5deg, pose_auc_5deg)
            best_pose_auc_5deg = pose_auc_5deg
            best_aSb = aSb_candidate

            aRb = best_aSb.rotation().matrix()
            atb = best_aSb.translation()
            rz, ry, rx = Rotation.from_matrix(aRb).as_euler("zyx", degrees=True)
            logger.debug("Sim(3) Rotation `aRb`: rz=%.2f deg., ry=%.2f deg., rx=%.2f deg.", rz, ry, rx)
            logger.debug("Sim(3) Translation `atb`: [%.2f, %.2f, %.2f]", atb[0], atb[1], atb[2])
            logger.debug("Sim(3) Scale `asb`: %.2f", float(best_aSb.scale()))

    aRb = best_aSb.rotation().matrix()
    atb = best_aSb.translation()
    rz, ry, rx = Rotation.from_matrix(aRb).as_euler("zyx", degrees=True)
    logger.info("Sim(3) Rotation `aRb`: rz=%.2f deg., ry=%.2f deg., rx=%.2f deg.", rz, ry, rx)
    logger.info("Sim(3) Translation `atb`: [%.2f, %.2f, %.2f]", atb[0], atb[1], atb[2])
    logger.info("Sim(3) Scale `asb`: %.2f", float(best_aSb.scale()))

    best_aTi_: List[Pose3] = [best_aSb.transformFrom(bTi) for bTi in bTi_list]
    return best_aTi_, best_aSb


def align_poses_sim3(aTi_list: List[Pose3], bTi_list: List[Pose3]) -> Tuple[List[Pose3], Similarity3]:
    """Align two pose graphs via similarity transformation. Note: poses cannot be missing/invalid.

    We force Sim(3) alignment rather than SE(3) alignment.
    We assume the two trajectories are of the exact same length.

    Args:
        aTi_list: reference poses in frame "a" which are the targets for alignment
        bTi_list: input poses which need to be aligned to frame "a"

    Returns:
        aTi_list_: transformed input poses previously "bTi_list" but now which
            have the same origin and scale as reference (now living in "a" frame)
        aSb: Similarity(3) object that aligns the two pose graphs.
    """
    assert len(aTi_list) == len(bTi_list)

    valid_pose_tuples = [
        pose_tuple
        for pose_tuple in list(zip(aTi_list, bTi_list))
        if pose_tuple[0] is not None and pose_tuple[1] is not None
    ]
    n_to_align = len(valid_pose_tuples)
    if n_to_align < 2:
        logger.error("SIM(3) alignment uses at least 2 frames; Skipping")
        return bTi_list, Similarity3(Rot3(), np.zeros((3,)), 1.0)

    ab_pairs = Pose3Pairs(valid_pose_tuples)

    aSb = Similarity3.Align(ab_pairs)

    if np.isnan(aSb.scale()) or aSb.scale() == 0:
        logger.warning("GTSAM Sim3.Align failed. Aligning ourselves")
        # we have run into a case where points have no translation between them (i.e. panorama).
        # We will first align the rotations and then align the translation by using centroids.
        # TODO: handle it in GTSAM

        # Align the rotations first, so that we can find the translation between the two panoramas.
        aSb = Similarity3(aSb.rotation(), np.zeros((3,)), 1.0)
        aTi_list_rot_aligned = [aSb.transformFrom(bTi) for _, bTi in valid_pose_tuples]

        # Fit a single translation motion to the centroid.
        aTi_centroid = np.array([aTi.translation() for aTi, _ in valid_pose_tuples]).mean(axis=0)
        aTi_rot_aligned_centroid = np.array([aTi.translation() for aTi in aTi_list_rot_aligned]).mean(axis=0)

        # Construct the final Sim(3) transform.
        aSb = Similarity3(aSb.rotation(), aTi_centroid - aTi_rot_aligned_centroid, 1.0)

    aSb = Similarity3(R=aSb.rotation(), t=aSb.translation(), s=aSb.scale())

    # Provide a summary of the estimated alignment transform.
    aRb = aSb.rotation().matrix()
    atb = aSb.translation()
    rz, ry, rx = Rotation.from_matrix(aRb).as_euler("zyx", degrees=True)
    logger.debug("Sim(3) Rotation `aRb`: rz=%.2f deg., ry=%.2f deg., rx=%.2f deg.", rz, ry, rx)
    logger.debug("Sim(3) Translation `atb`: [%.2f, %.2f, %.2f]", atb[0], atb[1], atb[2])
    logger.debug("Sim(3) Scale `asb`: %.2f", float(aSb.scale()))

    aTi_list_: List[Pose3] = []
    for bTi in bTi_list:
        aTi_list_.append(aSb.transformFrom(bTi))

    logger.debug("Pose graph Sim(3) alignment complete.")
    return aTi_list_, aSb


def align_gtsfm_data_via_Sim3_to_poses(input_data: GtsfmData, wTi_list_ref: List[Optional[Pose3]]) -> GtsfmData:
    """Align GtsfmData (points and cameras) to a set of reference poses.

    Args:
        wTi_list_ref: list of reference/target camera poses, ordered by camera index.

    Returns:
        aligned_data: GtsfmData that is aligned to the poses above.
    """
    # these are the estimated poses (source, to be aligned)
    wTi_list = input_data.get_camera_poses()
    # align the poses which are valid (i.e. are not None)
    # some camera indices may have been lost after pruning to largest connected component, leading to None values
    # rSe aligns the estimate `e` frame to the reference `r` frame
    _, rSe = align_poses_sim3_ignore_missing(wTi_list_ref, wTi_list)
    return input_data.apply_Sim3(aSb=rSe)
