"""Utilities to compute and save evaluation metrics.

Authors: Ayush Baid, Akshay Krishnan
"""
import itertools
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from dask.delayed import Delayed
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Point3, Pose3, Rot3, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.keypoints import Keypoints
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# A StatsDict is a dict from string to optional floats or their lists.
StatsDict = Dict[str, Union[Optional[float], List[Optional[float]]]]

METRICS_PATH = Path(__file__).resolve().parent.parent.parent / "result_metrics"
REACT_METRICS_PATH = Path(__file__).resolve().parent.parent.parent / "rtf_vis_tool" / "src" / "result_metrics"


logger = logger_utils.get_logger()


def count_correct_correspondences(
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    intrinsics_i1: Cal3Bundler,
    intrinsics_i2: Cal3Bundler,
    i2Ti1: Pose3,
    epipolar_dist_threshold: float,
) -> np.ndarray:
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
        Boolean mask of which verified correspondences are classified as correct under Sampson error
            (using GT epipolar geometry).
    """
    # TODO: add unit test, with mocking.
    if len(keypoints_i1) != len(keypoints_i2):
        raise ValueError("Keypoints must have same counts")

    if len(keypoints_i1) == 0:
        return 0

    i2Ei1 = EssentialMatrix(i2Ti1.rotation(), Unit3(i2Ti1.translation()))
    i2Fi1 = verification_utils.essential_to_fundamental_matrix(i2Ei1, intrinsics_i1, intrinsics_i2)

    distance_squared = verification_utils.compute_epipolar_distances_sq_sampson(
        keypoints_i1.coordinates, keypoints_i2.coordinates, i2Fi1
    )
    inlier_mask_gt = distance_squared < epipolar_dist_threshold ** 2
    return inlier_mask_gt


def compute_rotation_angle_metric(wRi_list: List[Optional[Rot3]], gt_wRi_list: List[Optional[Pose3]]) -> GtsfmMetric:
    """Computes statistics for the angle between estimated and GT rotations.

    Assumes that the estimated and GT rotations have been aligned and do not
    have a gauge freedom.

    Args:
        wRi_list: List of estimated camera rotations.
        gt_wRi_list: List of ground truth camera rotations.

    Returns:
        A statistics dict of the metrics errors in degrees.
    """
    errors = []
    for (wRi, gt_wRi) in zip(wRi_list, gt_wRi_list):
        if wRi is not None and gt_wRi is not None:
            errors.append(comp_utils.compute_relative_rotation_angle(wRi, gt_wRi))
    return GtsfmMetric("rotation_error_angle_deg", errors)


def compute_translation_distance_metric(
    wti_list: List[Optional[Point3]], gt_wti_list: List[Optional[Point3]]
) -> GtsfmMetric:
    """Computes statistics for the distance between estimated and GT translations.

    Assumes that the estimated and GT translations have been aligned and do not
    have a gauge freedom (including scale).

    Args:
        wti_list: List of estimated camera translations.
        gt_wti_list: List of ground truth camera translations.

    Returns:
        A statistics dict of the metrics errors in degrees.
    """
    errors = []
    for (wti, gt_wti) in zip(wti_list, gt_wti_list):
        if wti is not None and gt_wti is not None:
            errors.append(comp_utils.compute_points_distance_l2(wti, gt_wti))
    return GtsfmMetric("translation_error_distance", errors)


def compute_translation_angle_metric(
    i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]], wTi_list: List[Optional[Pose3]]
) -> GtsfmMetric:
    """Computes statistics for angle between translations and direction measurements.

    Args:
        i2Ui1_dict: List of translation direction measurements.
        wTi_list: List of estimated camera poses.

    Returns:
        A GtsfmMetric for the translation angle errors, in degrees.
    """
    angles = []
    for (i1, i2) in i2Ui1_dict:
        i2Ui1 = i2Ui1_dict[(i1, i2)]
        angles.append(comp_utils.compute_translation_to_direction_angle(i2Ui1, wTi_list[i2], wTi_list[i1]))
    return GtsfmMetric("translation_angle_error_deg", np.array(angles, dtype=np.float))


def compute_rotation_averaging_metrics(
    wRi_list: List[Optional[Rot3]],
    wti_list: List[Optional[Point3]],
    gt_wTi_list: List[Pose3],
) -> GtsfmMetricsGroup:
    """Computes statistics of multiple metrics for the averaging modules.

    Specifically, computes statistics of:
        - Rotation angle errors before BA,
        - Translation distances before BA,
        - Translation angle to direction measurements,

    Estimated poses and ground truth poses are first aligned before computing metrics.

    Args:
        wRi_list: List of estimated rotations.
        wti_list: List of estimated translations.
        gt_wTi_list: List of ground truth poses.

    Returns:
        A group of metrics that describe errors associated with an averaging result (w.r.t. GT).

    Raises:
        ValueError if lengths of wRi_list, wti_list and gt_wTi_list are not all same.
    """
    if len(wRi_list) != len(wti_list) or len(wRi_list) != len(gt_wTi_list):
        raise ValueError("Lengths of wRi_list, wti_list and gt_wTi_list should be the same.")

    wTi_list = []
    for (wRi, wti) in zip(wRi_list, wti_list):
        # if translation estimation failed in translation averaging, some wti_list values will be None
        if wRi is None or wti is None:
            wTi_list.append(None)
        else:
            wTi_list.append(Pose3(wRi, wti))

    # ground truth is the reference/target for alignment. discard 2nd return arg -- the estimated Similarity(3) object
    wTi_aligned_list, _ = comp_utils.align_poses_sim3_ignore_missing(gt_wTi_list, wTi_list)

    wRi_aligned_list, wti_aligned_list = get_rotations_translations_from_poses(wTi_aligned_list)
    gt_wRi_list, gt_wti_list = get_rotations_translations_from_poses(gt_wTi_list)

    metrics = []
    metrics.append(GtsfmMetric(name="num_rotations_computed", data=len([x for x in wRi_list if x is not None])))
    metrics.append(compute_rotation_angle_metric(wRi_aligned_list, gt_wRi_list))
    return GtsfmMetricsGroup(name="rotation_averaging_metrics", metrics=metrics)


def compute_ba_pose_metrics(
    gt_wTi_list: List[Pose3],
    ba_output: GtsfmData,
) -> GtsfmMetricsGroup:
    """Compute pose errors w.r.t. GT for the bundle adjustment result.

    Note: inputs must be aligned beforehand to the ground truth.

    Args:
        gt_wTi_list: List of ground truth poses.
        ba_output: sparse multi-view result, as output of bundle adjustment.

    Returns:
        A group of metrics that describe errors associated with a bundle adjustment result (w.r.t. GT).
    """
    wTi_aligned_list = ba_output.get_camera_poses()
    i2Ui1_dict_gt = get_twoview_translation_directions(gt_wTi_list)

    wRi_aligned_list, wti_aligned_list = get_rotations_translations_from_poses(wTi_aligned_list)
    gt_wRi_list, gt_wti_list = get_rotations_translations_from_poses(gt_wTi_list)

    metrics = []
    metrics.append(compute_rotation_angle_metric(wRi_aligned_list, gt_wRi_list))
    metrics.append(compute_translation_distance_metric(wti_aligned_list, gt_wti_list))
    metrics.append(compute_translation_angle_metric(i2Ui1_dict_gt, wTi_aligned_list))
    return GtsfmMetricsGroup(name="ba_pose_error_metrics", metrics=metrics)


def get_twoview_translation_directions(wTi_list: List[Pose3]) -> Dict[Tuple[int, int], Unit3]:
    """Generate synthetic measurements of the 2-view translation directions between image pairs.

    Args:
        wTi_list: List of poses (e.g. could be ground truth).

    Returns:
        i2Ui1_dict: Dict from (i1, i2) to unit translation direction i2Ui1.
    """
    number_images = len(wTi_list)  # vs. using ba_output.number_images()

    # check against all possible image pairs -- compute unit translation directions
    i2Ui1_dict = {}
    possible_img_pair_idxs = list(itertools.combinations(range(number_images), 2))
    for (i1, i2) in possible_img_pair_idxs:
        # compute the exact relative pose
        i2Ti1 = wTi_list[i2].between(wTi_list[i1])
        i2Ui1_dict[(i1, i2)] = Unit3(i2Ti1.translation())

    return i2Ui1_dict


def get_precision_recall_from_errors(
    positive_errors: List[float], negative_errors: List[float], max_positive_error: float
) -> Tuple[float, float]:
    """Computes the precision and recall from a list of errors for positive and negative classes.
    True positives are those for which the error is less than max_positive_error.

    Args:
        positive_errors: List of errors for the predicted positive instances.
        negative_errors: List of errors for the predicted negative instances.
        max_positive_error: Maximum error for a true positive prediction.

    Returns:
        Tuple of precision, recall.
    """
    tp = np.sum(np.array(positive_errors) <= max_positive_error)
    fp = np.sum(np.array(positive_errors) > max_positive_error)
    fn = np.sum(np.array(negative_errors) <= max_positive_error)

    eps = 1e-12  # prevent division by zero
    precision = tp * 1.0 / (tp + fp + eps)
    recall = tp * 1.0 / (tp + fn + eps)
    return precision, recall


def get_rotations_translations_from_poses(
    poses: List[Optional[Pose3]],
) -> Tuple[List[Optional[Rot3]], List[Optional[Point3]]]:
    """Decompose each 6-dof pose to a 3-dof rotation and 3-dof position"""
    rotations = []
    translations = []
    for pose in poses:
        if pose is None:
            rotations.append(None)
            translations.append(None)
            continue
        rotations.append(pose.rotation())
        translations.append(pose.translation())
    return rotations, translations


def save_metrics_as_json(metrics_groups: Delayed, output_dir: str) -> None:
    """Saves the input metrics groups as JSON files using the name of the group.

    Args:
        metrics_groups: List of GtsfmMetricsGroup to be saved.
        output_dir: Directory to save metrics to.
    """
    for metrics_group in metrics_groups:
        metrics_group.save_to_json(os.path.join(output_dir, metrics_group.name + ".json"))
