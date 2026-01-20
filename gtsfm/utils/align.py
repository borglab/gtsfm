"""Utility functions for aligning different geometry types.

Authors: Ayush Baid, John Lambert
"""

import copy
from typing import Optional, Sequence

import gtsam  # type: ignore
import numpy as np
from gtsam import LevenbergMarquardtOptimizer, NonlinearFactorGraph, Pose3, PriorFactorPose3, Rot3, Similarity3, Values
from gtsam.symbol_shorthand import X  # type: ignore

import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils

logger = logger_utils.get_logger()

MAX_NUM_HYPOTHESES_FOR_ROBUST_ALIGNMENT: int = 200

Z_3x1: np.ndarray = np.zeros((3,))

KEY = X(0)


def _log_sim3_transform(sim3: Similarity3, label: str = "Sim(3)") -> None:
    """Log rotation, translation, and scale components of a Similarity3."""
    aRb = sim3.rotation()
    atb = sim3.translation()
    rx, ry, rz = aRb.xyz()
    logger.debug(
        "%s Rotation `aRb`: rz=%.2f deg., ry=%.2f deg., rx=%.2f deg.",
        label,
        np.degrees(rz),
        np.degrees(ry),
        np.degrees(rx),
    )
    logger.debug("%s Translation `atb`: [%.2f, %.2f, %.2f]", label, atb[0], atb[1], atb[2])
    logger.debug("%s Scale `asb`: %.2f", label, float(sim3.scale()))


def _create_aTb_factors(a: dict[int, Pose3], b: dict[int, Pose3], common_keys: Sequence[int]) -> NonlinearFactorGraph:
    """Create a factor graph encoding pose priors for the unknown transform ``aTb``."""
    sigmas: np.ndarray = np.array([0.1] * 3 + [0.1] * 3)
    noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    graph = NonlinearFactorGraph()
    for key in common_keys:
        aTi = a[key]
        bTi = b[key]
        graph.add(PriorFactorPose3(KEY, aTi.compose(bTi.inverse()), noise_model))
    return graph


def _create_aTb_initial_estimate(a: dict[int, Pose3], b: dict[int, Pose3], common_keys: Sequence[int]) -> Values:
    """Seed the optimizer with an initial guess for ``aTb``."""
    initial = Values()
    first_key = common_keys[0]
    initial.insert(KEY, a[first_key].compose(b[first_key].inverse()))
    return initial


def so3_from_optional_Rot3s(aRi_list: Sequence[Optional[Rot3]], bRi_list: Sequence[Optional[Rot3]]) -> Rot3:
    """Align the list of rotations to the reference list using the Karcher mean."""
    aRb_list = [aRi * bRi.inverse() for aRi, bRi in zip(aRi_list, bRi_list) if aRi is not None and bRi is not None]
    return gtsam.FindKarcherMeanRot3(aRb_list) if len(aRb_list) > 0 else Rot3()


def so3_from_Pose3s(aTi_list: Sequence[Pose3], bTi_list: Sequence[Pose3]) -> Rot3:
    """Align the list of rotations to the reference list using the Karcher mean."""
    aRb_list = [
        aTi.rotation() * bTi.rotation().inverse()
        for aTi, bTi in zip(aTi_list, bTi_list)
        if aTi is not None and bTi is not None
    ]
    return gtsam.FindKarcherMeanRot3(aRb_list) if len(aRb_list) > 0 else Rot3()


def se3_from_Pose3_maps(a: dict[int, Pose3], b: dict[int, Pose3]) -> Pose3:
    """Estimate the SE(3) transform ``aTb`` that best aligns the overlapping poses."""
    common_keys = [key for key in a if key in b]
    if not common_keys:
        raise ValueError("No overlapping cameras found between the pose dictionaries.")

    graph = _create_aTb_factors(a, b, common_keys)
    initial = _create_aTb_initial_estimate(a, b, common_keys)
    try:
        optimizer = LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()
    except Exception as exc:  # pragma: no cover - GTSAM errors contain rich debugging info.
        raise RuntimeError(f"GTSAM optimization failed during pose alignment: {exc}") from exc
    return result.atPose3(KEY)


def sim3_from_Pose3_maps(a: dict[int, Pose3], b: dict[int, Pose3]) -> Similarity3:
    """Estimate the Sim(3) transform ``aSb`` that best aligns the overlapping poses."""
    common_keys = [key for key in a if key in b]
    pose_pairs = [(a[key], b[key]) for key in common_keys]
    if not pose_pairs:
        raise ValueError("No overlapping cameras found between the pose dictionaries.")
    try:
        return Similarity3.Align(pose_pairs)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"align.sim3_from_Pose3_maps: Similarity3.Align failed with {len(pose_pairs)} pose pairs: {exc}."
        ) from exc


def sim3_from_Pose3_maps_robust(aTi: dict[int, Pose3], bTi: dict[int, Pose3]) -> Similarity3:
    """Estimate Sim(3) alignment while allowing missing poses in the inputs.

    This is a convenience wrapper for ``estimate_sim3_robust`` that tolerates dropped cameras.
    We force Sim(3) alignment rather than SE(3) alignment and assume the two trajectories are of
    the exact same length.

    Args:
        aTi_list: Reference poses in frame "a" which are the targets for alignment.
        bTi_list: Input poses which need to be aligned to frame "a".

    Returns:
        aSb: Similarity(3) object that aligns the two pose graphs.
    """
    # Only choose target poses for which there is a corresponding estimated pose.
    corresponding_aTi = []
    corresponding_bTi = []
    for i, aTi in aTi.items():
        if aTi is not None and bTi.get(i) is not None:
            corresponding_aTi.append(aTi)
            corresponding_bTi.append(bTi.get(i))

    return sim3_from_Pose3s_robust(aTi_list=corresponding_aTi, bTi_list=corresponding_bTi)


def sim3_from_Pose3s_exhaustive(aTi_list: list[Pose3], bTi_list: list[Pose3]) -> Similarity3:
    """Estimate Sim(3) alignment by exhaustively sampling pose pairs.

    Poses cannot be missing or invalid. We force Sim(3) alignment rather than SE(3) alignment and
    assume the two trajectories are of the exact same length.

    Args:
        aTi_list: Reference poses in frame "a" which are the targets for alignment.
        bTi_list: Input poses which need to be aligned to frame "a".

    Returns:
        aSb: Similarity(3) object that aligns the two pose graphs.
    """
    assert len(aTi_list) == len(bTi_list)
    n_to_align = len(aTi_list)
    if n_to_align < 2:
        logger.error("SIM(3) alignment uses at least 2 frames; Skipping")
        return Similarity3(Rot3(), Z_3x1, 1.0)

    # Run once with all poses for initial guess.
    best_aSb = sim3_from_Pose3s(aTi_list, bTi_list)
    aTi_candidate_all: list[Pose3] = [best_aSb.transformFrom(bTi) for bTi in bTi_list]
    best_pose_auc_5deg: float = metric_utils.pose_auc_from_poses(
        computed_wTis={i: pose for i, pose in enumerate(aTi_candidate_all)},
        ref_wTis={i: pose for i, pose in enumerate(aTi_list)},
        thresholds_deg=[5],
    )[0]

    for i1 in range(n_to_align):
        for i2 in range(i1 + 1, n_to_align):
            aTi_sample = copy.deepcopy([aTi_list[i1], aTi_list[i2]])
            bTi_sample = copy.deepcopy([bTi_list[i1], bTi_list[i2]])

            aSb_candidate = sim3_from_Pose3s(aTi_sample, bTi_sample)

            aTi_candidate_: list[Pose3] = [aSb_candidate.transformFrom(bTi) for bTi in bTi_list]

            pose_auc_5deg = metric_utils.pose_auc_from_poses(
                computed_wTis={i: pose for i, pose in enumerate(aTi_candidate_)},
                ref_wTis={i: pose for i, pose in enumerate(aTi_list)},
                thresholds_deg=[5],
            )[0]

            if pose_auc_5deg > best_pose_auc_5deg:
                logger.debug("Update auc: %.2f -> %.2f", best_pose_auc_5deg, pose_auc_5deg)
                best_pose_auc_5deg = pose_auc_5deg
                best_aSb = aSb_candidate

                _log_sim3_transform(best_aSb)

    _log_sim3_transform(best_aSb)

    return best_aSb


def sim3_from_Pose3s_robust(
    aTi_list: list[Pose3], bTi_list: list[Pose3], max_num_hypotheses: int = MAX_NUM_HYPOTHESES_FOR_ROBUST_ALIGNMENT
) -> Similarity3:
    """Estimate Sim(3) alignment using random pose pair sampling for robustness.

    Poses cannot be missing or invalid. We force Sim(3) alignment rather than SE(3) alignment and
    assume the two trajectories are of the exact same length.

    Args:
        aTi_list: Reference poses in frame "a" which are the targets for alignment.
        bTi_list: Input poses which need to be aligned to frame "a".
        max_num_hypothesis: max number of RANSAC iterations.

    Returns:
        aSb: Similarity(3) object that aligns the two pose graphs.
    """
    assert len(aTi_list) == len(bTi_list)
    n_to_align = len(aTi_list)
    if n_to_align < 2:
        logger.error("SIM(3) alignment uses at least 2 frames; Skipping")
        return Similarity3(Rot3(), Z_3x1, 1.0)

    # Compute the total possible number of hypothesis { N choose 2 }
    max_possible_hypotheses: int = (n_to_align * (n_to_align - 1)) // 2
    if max_possible_hypotheses <= max_num_hypotheses:
        return sim3_from_Pose3s_exhaustive(aTi_list=aTi_list, bTi_list=bTi_list)

    # Run once with all poses for initial guess
    best_aSb = Similarity3()
    best_aSb = sim3_from_Pose3s(aTi_list, bTi_list)
    aTi_candidate_all: list[Pose3] = [best_aSb.transformFrom(bTi) for bTi in bTi_list]
    best_pose_auc_5deg: float = metric_utils.pose_auc_from_poses(
        computed_wTis={i: pose for i, pose in enumerate(aTi_candidate_all)},
        ref_wTis={i: pose for i, pose in enumerate(aTi_list)},
        thresholds_deg=[5],
    )[0]

    sample_pose_pairs = np.random.choice(
        len(aTi_list),
        size=max_num_hypotheses * 2,
        replace=True,
    ).reshape(-1, 2)

    for i1, i2 in sample_pose_pairs:
        aTi_sample = copy.deepcopy([aTi_list[i1], aTi_list[i2]])
        bTi_sample = copy.deepcopy([bTi_list[i1], bTi_list[i2]])

        aSb_candidate = sim3_from_Pose3s(aTi_sample, bTi_sample)

        aTi_candidate_: list[Pose3] = [aSb_candidate.transformFrom(bTi) for bTi in bTi_list]

        pose_auc_5deg = metric_utils.pose_auc_from_poses(
            computed_wTis={i: pose for i, pose in enumerate(aTi_candidate_)},
            ref_wTis={i: pose for i, pose in enumerate(aTi_list)},
            thresholds_deg=[5],
        )[0]

        if pose_auc_5deg > best_pose_auc_5deg:
            logger.debug("Update auc: %.2f -> %.2f", best_pose_auc_5deg, pose_auc_5deg)
            best_pose_auc_5deg = pose_auc_5deg
            best_aSb = aSb_candidate

            _log_sim3_transform(best_aSb)

    _log_sim3_transform(best_aSb)

    return best_aSb


def sim3_from_Pose3s(aTi_list: list[Pose3], bTi_list: list[Pose3]) -> Similarity3:
    """Estimate Sim(3) alignment between two pose graphs.

    Poses cannot be missing or invalid.
    We force Sim(3) alignment rather than SE(3) alignment and assume the two trajectories are of the exact
    same length.

    Args:
        aTi_list: Reference poses in frame "a" which are the targets for alignment
        bTi_list: Input poses which need to be aligned to frame "a"

    Returns:
        aSb: Similarity(3) object that aligns the two pose graphs.
    """
    assert len(aTi_list) == len(bTi_list)

    valid_pose_tuples = [
        (aTi, bTi) for (aTi, bTi) in list(zip(aTi_list, bTi_list)) if aTi is not None and bTi is not None
    ]
    n_to_align = len(valid_pose_tuples)
    if n_to_align < 2:
        logger.error("SIM(3) alignment uses at least 2 frames; Skipping")
        return Similarity3(Rot3(), Z_3x1, 1.0)

    aSb = Similarity3.Align(valid_pose_tuples)
    if np.isnan(aSb.scale()) or aSb.scale() == 0:
        logger.debug("GTSAM Sim3.Align failed with scale: %.2f. Aligning ourselves", aSb.scale())
        # we have run into a case where points have no translation between them (i.e. panorama).
        # We will first align the rotations and then align the translation by using centroids.
        # TODO: handle it in GTSAM
        # TODO(Frank): It does not only fail for panoramas! (otherwise translation calculation would be zero)

        # Align the rotations first, so that we can find the translation between the two panoramas.
        aSb = Similarity3(aSb.rotation(), Z_3x1, 1.0)
        aTi_list_rot_aligned = [aSb.transformFrom(bTi) for _, bTi in valid_pose_tuples]

        # Fit a single translation motion to the centroid.
        aTi_centroid = np.array([aTi.translation() for aTi, _ in valid_pose_tuples]).mean(axis=0)
        aTi_rot_aligned_centroid = np.array([aTi.translation() for aTi in aTi_list_rot_aligned]).mean(axis=0)

        # Construct the final Sim(3) transform.
        aSb = Similarity3(aSb.rotation(), aTi_centroid - aTi_rot_aligned_centroid, 1.0)

    # Provide a summary of the estimated alignment transform.
    _log_sim3_transform(aSb)

    logger.debug("Pose graph Sim(3) alignment complete.")
    return aSb
