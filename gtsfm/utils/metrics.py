"""Utilities to compute and save evaluation metrics.

Authors: Ayush Baid, Akshay Krishnan
"""
import timeit
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
from gtsam import Cal3Bundler, EssentialMatrix, Point3, Pose3, Rot3, Unit3
import matplotlib.pyplot as plt

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.keypoints import Keypoints

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# A StatsDict is a dict from string to optional floats or their lists.
StatsDict = Dict[str, Union[Optional[float], List[Optional[float]]]]

# number of digits (significant figures) to include in each entry of error metrics
PRINT_NUM_SIG_FIGS = 2


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
        Mask of inlier correspondences.
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
    return np.array(distance_squared < epipolar_dist_threshold ** 2)


def mesh_inlier_correspondences(
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    camera_intrinsics_i1: Cal3Bundler,
    camera_intrinsics_i2: Cal3Bundler,
    gt_wTi1: Pose3,
    gt_wTi2: Pose3,
    gt_scene_mesh: trimesh.Trimesh,
) -> Tuple[np.ndarray, Optional[float]]:
    """Compute inlier correspondences using the ground truth triangular surface mesh of the scene.

    Args:
        keypoints_i1: N keypoints in image i1.
        keypoints_i2: N corresponding keypoints in image i2.
        intrinsics_i1: intrinsics for i1.
        intrinsics_i2: intrinsics for i2.
        gt_wTi1: ground truth pose of the world frame relative to i1.
        gt_wTi2: ground truth pose of the world frame relative to i2.
        gt_scene_mesh: ground truth triangular surface mesh of the scene in the world frame.

    Raises:
        ValueError: when the number of keypoints do not match.

    Returns:
        is_inlier: (N, ) mask of inlier correspondences.
    """
    def back_project(u, v, fx, fy, cx, cy, wRi: Rot3):            
        """Back-project ray from pixel coord"""
        zhat = (1 + (u - cx)**2/fx**2 + (v - cy)**2/fy**2)**(-1/2)
        xhat = zhat/fx*(u - cx)
        yhat = zhat/fy*(v - cy)
        return np.dot(wRi.matrix(), np.array([[xhat], [yhat], [zhat]]))

    def forward_project(wtlw: np.ndarray, fx, fy, cx, cy, iTw: Pose3):            
        itwi = np.reshape(iTw.translation(), (3, 1))
        wtlw = np.reshape(wtlw, (3, 1))
        itli = np.dot(iTw.rotation().matrix(), wtlw) + itwi    
        x, y, z = itli
        return np.array([fx/z*x + cx, fy/z*y + cy])

    # TODO: add unit test, with mocking.
    if len(keypoints_i1) != len(keypoints_i2):
        raise ValueError("Keypoints must have same counts")

    fx_i1, fy_i1, cx_i1, cy_i1 = camera_intrinsics_i1.fx(), camera_intrinsics_i1.fy(), camera_intrinsics_i1.px(), camera_intrinsics_i1.py()
    fx_i2, fy_i2, cx_i2, cy_i2 = camera_intrinsics_i2.fx(), camera_intrinsics_i2.fy(), camera_intrinsics_i2.px(), camera_intrinsics_i2.py()
    n_corrs = len(keypoints_i1)
    is_inlier = np.zeros(n_corrs, dtype=bool)
    src_i1 = np.repeat(np.reshape(gt_wTi1.translation(), (-1, 3)), n_corrs, axis=0) 
    src_i2 = np.repeat(np.reshape(gt_wTi2.translation(), (-1, 3)), n_corrs, axis=0) 

    # compute ketpoint rays
    drc_i1 = np.empty((n_corrs, 3), dtype=float)
    drc_i2 = np.empty((n_corrs, 3), dtype=float)
    for corr_idx in range(n_corrs):
        x_i1, y_i1 = keypoints_i1.coordinates[corr_idx]
        x_i2, y_i2 = keypoints_i2.coordinates[corr_idx]
        drc_i1[corr_idx, :] = np.reshape(back_project(x_i1, y_i1, fx_i1, fy_i1, cx_i1, cy_i1, gt_wTi1.rotation()), (-1, 3))
        drc_i2[corr_idx, :] = np.reshape(back_project(x_i2, y_i2, fx_i2, fy_i2, cx_i2, cy_i2, gt_wTi2.rotation()), (-1, 3))

    # perform ray tracing
    src = np.vstack((src_i1, src_i2))
    drc = np.vstack((drc_i1, drc_i2))
    #logger.info(f'Computing ray intersections...')
    _start = timeit.default_timer()
    loc, idr, _ = gt_scene_mesh.ray.intersects_location(src, drc, multiple_hits=False)
    _end = timeit.default_timer()
    #logger.info(f'Cast {2 * n_corrs} rays in {_end - _start} seconds.')

    # unpack results
    idr_i1 = idr[idr < n_corrs]
    loc_i1 = loc[idr < n_corrs]
    idr_i2 = idr[idr >= n_corrs]-n_corrs
    loc_i2 = loc[idr >= n_corrs]
    idr, i1_idx, i2_idx = np.intersect1d(idr_i1, idr_i2, return_indices=True)

    # forward project intersections into other image to compute error
    reproj_err = []
    for i in range(len(idr)):
        x_i1, y_i1 = keypoints_i1.coordinates[idr[i]]
        x_i2, y_i2 = keypoints_i2.coordinates[idr[i]]
        x_i2i1, y_i2i1 = forward_project(loc_i2[i2_idx[i]], fx_i1, fy_i1, cx_i1, cy_i1, gt_wTi1.inverse())
        x_i1i2, y_i1i2 = forward_project(loc_i1[i1_idx[i]], fx_i2, fy_i2, cx_i2, cy_i2, gt_wTi2.inverse())
        err_i2i1 = ((x_i1 - x_i2i1)**2 + (y_i1 - y_i2i1)**2)**0.5 # pixels
        err_i1i2 = ((x_i2 - x_i1i2)**2 + (y_i2 - y_i1i2)**2)**0.5 # pixels
        is_inlier[idr[i]] = max(err_i2i1, err_i1i2) < 10
        if is_inlier[idr[i]]:
            reproj_err.append(max(err_i2i1, err_i1i2))

    return is_inlier, np.mean(reproj_err)


def compute_errors_statistics(errors: List[Optional[float]]) -> StatsDict:
    """Computes statistics (min, max, median) on the given list of errors

    Args:
        errors: List of errors for a metric.

    Returns:
        A dict with keys min_error, max_error, median_error,
        and errors_list mapping to the respective stats.
    """
    metrics = {}
    valid_errors = [error for error in errors if error is not None]
    metrics["median_error"] = np.round(np.median(valid_errors), PRINT_NUM_SIG_FIGS)
    metrics["min_error"] = np.round(np.min(valid_errors), PRINT_NUM_SIG_FIGS)
    metrics["max_error"] = np.round(np.max(valid_errors), PRINT_NUM_SIG_FIGS)
    metrics["errors_list"] = [np.round(error, PRINT_NUM_SIG_FIGS) if error is not None else None for error in errors]
    return metrics


def compute_rotation_angle_metrics(wRi_list: List[Optional[Rot3]], gt_wRi_list: List[Optional[Pose3]]) -> StatsDict:
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
        errors.append(comp_utils.compute_relative_rotation_angle(wRi, gt_wRi))
    return compute_errors_statistics(errors)


def compute_translation_distance_metrics(
    wti_list: List[Optional[Point3]], gt_wti_list: List[Optional[Point3]]
) -> StatsDict:
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
        errors.append(comp_utils.compute_points_distance_l2(wti, gt_wti))
    return compute_errors_statistics(errors)


def compute_translation_angle_metrics(
    i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]], wTi_list: List[Optional[Pose3]]
) -> StatsDict:
    """Computes statistics for angle between translations and direction measurements.

    Args:
        i2Ui1_dict: List of translation direction measurements.
        wTi_list: List of estimated camera poses.

    Returns:
        A statistics dict of the metrics errors in degrees.
    """
    angles = []
    for (i1, i2) in i2Ui1_dict:
        i2Ui1 = i2Ui1_dict[(i1, i2)]
        angles.append(comp_utils.compute_translation_to_direction_angle(i2Ui1, wTi_list[i2], wTi_list[i1]))
    return compute_errors_statistics(angles)


def compute_averaging_metrics(
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    wRi_list: List[Optional[Rot3]],
    wti_list: List[Optional[Point3]],
    gt_wTi_list: List[Optional[Pose3]],
) -> Dict[str, StatsDict]:
    """Computes statistics of multiple metrics for the averaging modules.

    Specifically, computes statistics of:
        - Rotation angle errors before BA,
        - Translation distances before BA,
        - Translation angle to direction measurements,

    Estimated poses and ground truth poses are first aligned before computing metrics.

    Args:
        i2Ui1_dict: Dict from (i1, i2) to unit translation measurement i2Ui1.
        wRi_list: List of estimated rotations.
        wti_list: List of estimated translations.
        gt_wTi_list: List of ground truth poses.

    Returns:
        Dict from metric name to a StatsDict.

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

    # ground truth is the reference/target for alignment
    wTi_aligned_list = comp_utils.align_poses_sim3_ignore_missing(gt_wTi_list, wTi_list)

    wRi_aligned_list, wti_aligned_list = get_rotations_translations_from_poses(wTi_aligned_list)
    gt_wRi_list, gt_wti_list = get_rotations_translations_from_poses(gt_wTi_list)

    metrics = {}
    metrics["rotation_averaging_angle_deg"] = compute_rotation_angle_metrics(wRi_aligned_list, gt_wRi_list)
    metrics["translation_averaging_distance"] = compute_translation_distance_metrics(wti_aligned_list, gt_wti_list)
    metrics["translation_to_direction_angle_deg"] = compute_translation_angle_metrics(i2Ui1_dict, wTi_aligned_list)
    return metrics


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


def compute_pose_errors(gt_wTi_list: List[Pose3], wTi_list: List[Pose3]) -> Dict[str, StatsDict]:
    """Compare orientation and location errors for each estimated poses, vs. ground truth.

    Note: Poses must be aligned, before calling this function
    """
    wRi_list, wti_list = get_rotations_translations_from_poses(wTi_list)
    gt_wRi_list, gt_wti_list = get_rotations_translations_from_poses(gt_wTi_list)

    metrics = {}
    metrics["rotation_angle_deg_errors"] = compute_rotation_angle_metrics(wRi_list, gt_wRi_list)
    metrics["translation_distance_errors"] = compute_translation_distance_metrics(wti_list, gt_wti_list)
    return metrics


def log_sfm_summary() -> None:
    """Dump to stdout a summary of metrics about the SfM reconstruction process."""
    frontend_full_metrics_fpath = REPO_ROOT / "result_metrics" / "frontend_full.json"
    frontend_metrics = io_utils.read_json_file(frontend_full_metrics_fpath)

    rot_errs_deg = [
        pair_stats["rotation_angular_error"] for pair_stats in frontend_metrics if pair_stats["rotation_angular_error"]
    ]
    trans_errs_deg = [
        pair_stats["translation_angular_error"]
        for pair_stats in frontend_metrics
        if pair_stats["translation_angular_error"]
    ]

    logger.info("=============> Metrics report ==============>")
    logger.info("Front-end median_rot_err_deg: %.2f", np.median(rot_errs_deg))
    logger.info("Front-end max_rot_err_deg: %.2f", max(rot_errs_deg))

    logger.info("Front-end median_trans_err_deg: %.2f", np.median(trans_errs_deg))
    logger.info("Front-end max_trans_err_deg: %.2f", max(trans_errs_deg))
    
    averaging_metrics_fpath = REPO_ROOT / "result_metrics" / "averaging_metrics.json"
    averaging_metrics = io_utils.read_json_file(averaging_metrics_fpath)

    logger.info("Averaging median_rot_err_deg: %.2f", averaging_metrics["rotation_averaging_angle_deg"]["median_error"])
    logger.info("Averaging max_rot_err_deg: %.2f", averaging_metrics["rotation_averaging_angle_deg"]["max_error"])

    logger.info(
        "Averaging median_trans_dist_err: %.2f", averaging_metrics["translation_averaging_distance"]["median_error"]
    )
    logger.info("Averaging max_trans_dist_err: %.2f", averaging_metrics["translation_averaging_distance"]["max_error"])
