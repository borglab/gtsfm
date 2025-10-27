"""Utility functions for aligning different geometry types.

Authors: Ayush Baid, John Lambert
"""

import copy
from typing import List, Optional, Sequence, Tuple

import gtsam  # type: ignore
import numpy as np
from gtsam import Pose3, Pose3Pairs, Rot3, Similarity3

import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils
from gtsfm.common.gtsfm_data import GtsfmData

EPSILON = np.finfo(float).eps

logger = logger_utils.get_logger()

MAX_ALIGNMENT_ERROR = np.finfo(np.float32).max
MAX_NUM_HYPOTHESES_FOR_ROBUST_ALIGNMENT: int = 200

Z_3x1: np.ndarray = np.zeros((3,))


def log_sim3_transform(sim3: Similarity3, label: str = "Sim(3)") -> None:
    """Log rotation, translation, and scale components of a Similarity3."""
    aRb = sim3.rotation()
    atb = sim3.translation()
    rx, ry, rz = aRb.xyz()
    logger.debug("%s Rotation `aRb`: rz=%.2f deg., ry=%.2f deg., rx=%.2f deg.", label, rz, ry, rx)
    logger.debug("%s Translation `atb`: [%.2f, %.2f, %.2f]", label, atb[0], atb[1], atb[2])
    logger.debug("%s Scale `asb`: %.2f", label, float(sim3.scale()))


def align_rotations(aRi_list: Sequence[Optional[Rot3]], bRi_list: Sequence[Optional[Rot3]]) -> List[Optional[Rot3]]:
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
        aRb = gtsam.FindKarcherMeanRot3(aRb_list)
    else:
        aRb = Rot3()

    # Apply the coordinate shift to all entries in input.
    return [aRb.compose(bRi) if bRi is not None else None for bRi in bRi_list]


def align_poses_sim3_ignore_missing(
    aTi_list: Sequence[Optional[Pose3]], bTi_list: Sequence[Optional[Pose3]]
) -> Tuple[List[Optional[Pose3]], Similarity3]:
    """Align by similarity transformation, but allow missing estimated poses in the input.

    Note: this is a wrapper for align_poses_sim3() that allows for missing poses/dropped cameras.
    This is necessary, as align_poses_sim3() requires a valid pose for every input pair.

    We force SIM(3) alignment rather than SE(3) alignment.
    We assume the two trajectories are of the exact same length.

    Args:
        aTi_list: Reference poses in frame "a" which are the targets for alignment.
        bTi_list: Input poses which need to be aligned to frame "a".

    Returns:
        aTi_list_: Transformed input poses previously "bTi_list" but now which
            have the same origin and scale as reference (now living in "a" frame)
        aSb: Similarity(3) object that aligns the two pose graphs.
    """
    assert len(aTi_list) == len(bTi_list)

    # Only choose target poses for which there is a corresponding estimated pose.
    corresponding_aTi_list = []
    valid_camera_idxs = []
    valid_bTi_list = []
    for i, bTi in enumerate(bTi_list):
        aTi = aTi_list[i]
        if aTi is None or bTi is None:
            continue
        valid_camera_idxs.append(i)
        valid_bTi_list.append(bTi)
        corresponding_aTi_list.append(aTi)

    valid_aTi_list_, aSb = align_poses_sim3_robust(aTi_list=corresponding_aTi_list, bTi_list=valid_bTi_list)

    num_cameras = len(aTi_list)
    # now at valid indices
    aTi_list_: List[Optional[Pose3]] = [None] * num_cameras
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
        aTi_list: Reference poses in frame "a" which are the targets for alignment.
        bTi_list: Input poses which need to be aligned to frame "a".

    Returns:
        aTi_list_: Transformed input poses previously "bTi_list" but now which
            have the same origin and scale as reference (now living in "a" frame)
        aSb: Similarity(3) object that aligns the two pose graphs.
    """
    assert len(aTi_list) == len(bTi_list)
    n_to_align = len(aTi_list)
    if n_to_align < 2:
        logger.error("SIM(3) alignment uses at least 2 frames; Skipping")
        return bTi_list, Similarity3(Rot3(), Z_3x1, 1.0)

    # Run once with all poses for initial guess.
    best_aSb = Similarity3()
    aTi_candidate_all, best_aSb = align_poses_sim3(aTi_list, bTi_list)
    best_pose_auc_5deg: float = metric_utils.pose_auc_from_poses(
        computed_wTis=aTi_candidate_all, ref_wTis=aTi_list, thresholds_deg=[5]
    )[0]

    for i1 in range(n_to_align):
        for i2 in range(i1 + 1, n_to_align):
            aTi_sample = copy.deepcopy([aTi_list[i1], aTi_list[i2]])
            bTi_sample = copy.deepcopy([bTi_list[i1], bTi_list[i2]])

            _, aSb_candidate = align_poses_sim3(aTi_sample, bTi_sample)

            aTi_candidate_: List[Pose3] = [aSb_candidate.transformFrom(bTi) for bTi in bTi_list]

            pose_auc_5deg = metric_utils.pose_auc_from_poses(
                computed_wTis=aTi_candidate_, ref_wTis=aTi_list, thresholds_deg=[5]
            )[0]

            if pose_auc_5deg > best_pose_auc_5deg:
                logger.debug("Update auc: %.2f -> %.2f", best_pose_auc_5deg, pose_auc_5deg)
                best_pose_auc_5deg = pose_auc_5deg
                best_aSb = aSb_candidate

                log_sim3_transform(best_aSb)

    log_sim3_transform(best_aSb)

    best_aTi_: List[Pose3] = [best_aSb.transformFrom(bTi) for bTi in bTi_list]
    return best_aTi_, best_aSb


def align_poses_sim3_robust(
    aTi_list: List[Pose3], bTi_list: List[Pose3], max_num_hypotheses: int = MAX_NUM_HYPOTHESES_FOR_ROBUST_ALIGNMENT
) -> Tuple[List[Pose3], Similarity3]:
    """Align two pose graphs using similarity transform chosen from pose pair samples.

    Note: poses cannot be missing/invalid.
    We force Sim(3) alignment rather than SE(3) alignment.
    We assume the two trajectories are of the exact same length.

    Args:
        aTi_list: Reference poses in frame "a" which are the targets for alignment.
        bTi_list: Input poses which need to be aligned to frame "a".
        max_num_hypothesis: max number of RANSAC iterations.

    Returns:
        aTi_list_: Transformed input poses previously "bTi_list" but now which
            have the same origin and scale as reference (now living in "a" frame).
        aSb: Similarity(3) object that aligns the two pose graphs.
    """
    assert len(aTi_list) == len(bTi_list)
    n_to_align = len(aTi_list)
    if n_to_align < 2:
        logger.error("SIM(3) alignment uses at least 2 frames; Skipping")
        return bTi_list, Similarity3(Rot3(), Z_3x1, 1.0)

    # Compute the total possible number of hypothesis { N choose 2 }
    max_possible_hypotheses: int = (n_to_align * (n_to_align - 1)) // 2
    if max_possible_hypotheses <= max_num_hypotheses:
        return align_poses_sim3_exhaustive(aTi_list=aTi_list, bTi_list=bTi_list)

    # Run once with all poses for initial guess
    best_aSb = Similarity3()
    aTi_candidate_all, best_aSb = align_poses_sim3(aTi_list, bTi_list)
    best_pose_auc_5deg: float = metric_utils.pose_auc_from_poses(
        computed_wTis=aTi_candidate_all, ref_wTis=aTi_list, thresholds_deg=[5]
    )[0]

    sample_pose_pairs = np.random.choice(
        len(aTi_list),
        size=max_num_hypotheses * 2,
        replace=True,
    ).reshape(-1, 2)

    for i1, i2 in sample_pose_pairs:
        aTi_sample = copy.deepcopy([aTi_list[i1], aTi_list[i2]])
        bTi_sample = copy.deepcopy([bTi_list[i1], bTi_list[i2]])

        _, aSb_candidate = align_poses_sim3(aTi_sample, bTi_sample)

        aTi_candidate_: List[Pose3] = [aSb_candidate.transformFrom(bTi) for bTi in bTi_list]

        pose_auc_5deg = metric_utils.pose_auc_from_poses(
            computed_wTis=aTi_candidate_, ref_wTis=aTi_list, thresholds_deg=[5]
        )[0]

        if pose_auc_5deg > best_pose_auc_5deg:
            logger.debug("Update auc: %.2f -> %.2f", best_pose_auc_5deg, pose_auc_5deg)
            best_pose_auc_5deg = pose_auc_5deg
            best_aSb = aSb_candidate

            log_sim3_transform(best_aSb)

    log_sim3_transform(best_aSb)

    best_aTi_: List[Pose3] = [best_aSb.transformFrom(bTi) for bTi in bTi_list]
    return best_aTi_, best_aSb


def align_poses_sim3(aTi_list: List[Pose3], bTi_list: List[Pose3]) -> Tuple[List[Pose3], Similarity3]:
    """Align two pose graphs via similarity transformation. Note: poses cannot be missing/invalid.

    We force Sim(3) alignment rather than SE(3) alignment.
    We assume the two trajectories are of the exact same length.

    Args:
        aTi_list: Reference poses in frame "a" which are the targets for alignment
        bTi_list: Input poses which need to be aligned to frame "a"

    Returns:
        aTi_list_: Transformed input poses previously "bTi_list" but now which
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
        return bTi_list, Similarity3(Rot3(), Z_3x1, 1.0)

    ab_pairs = Pose3Pairs(valid_pose_tuples)

    aSb = Similarity3.Align(ab_pairs)
    if np.isnan(aSb.scale()) or aSb.scale() == 0:
        logger.warning("GTSAM Sim3.Align failed. Aligning ourselves")
        # we have run into a case where points have no translation between them (i.e. panorama).
        # We will first align the rotations and then align the translation by using centroids.
        # TODO: handle it in GTSAM

        # Align the rotations first, so that we can find the translation between the two panoramas.
        aSb = Similarity3(aSb.rotation(), Z_3x1, 1.0)
        aTi_list_rot_aligned = [aSb.transformFrom(bTi) for _, bTi in valid_pose_tuples]

        # Fit a single translation motion to the centroid.
        aTi_centroid = np.array([aTi.translation() for aTi, _ in valid_pose_tuples]).mean(axis=0)
        aTi_rot_aligned_centroid = np.array([aTi.translation() for aTi in aTi_list_rot_aligned]).mean(axis=0)

        # Construct the final Sim(3) transform.
        aSb = Similarity3(aSb.rotation(), aTi_centroid - aTi_rot_aligned_centroid, 1.0)

    aSb = Similarity3(R=aSb.rotation(), t=aSb.translation(), s=aSb.scale())

    # Provide a summary of the estimated alignment transform.
    log_sim3_transform(aSb)

    aTi_list_: List[Pose3] = []
    for bTi in bTi_list:
        aTi_list_.append(aSb.transformFrom(bTi))

    logger.debug("Pose graph Sim(3) alignment complete.")
    return aTi_list_, aSb


def align_gtsfm_data_via_Sim3_to_poses(input_data: GtsfmData, wTi_list_ref: List[Optional[Pose3]]) -> GtsfmData:
    """Align GtsfmData (points and cameras) to a set of reference poses.

    Args:
        wTi_list_ref: List of reference/target camera poses, ordered by camera index.

    Returns:
        aligned_data: GtsfmData that is aligned to the poses above.
    """
    return input_data.aligned_via_sim3_to_poses(wTi_list_ref)
