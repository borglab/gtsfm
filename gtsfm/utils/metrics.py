"""Utilities to compute and save evaluation metrics.

Authors: Ayush Baid, Akshay Krishnan
"""
import itertools
import os
import timeit
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, Unit3
from trimesh import Trimesh

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

EPSILON = 1e-12

logger = logger_utils.get_logger()


def compute_correspondence_metrics(
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    corr_idxs_i1i2: np.ndarray,
    intrinsics_i1: Cal3Bundler,
    intrinsics_i2: Cal3Bundler,
    dist_threshold: float,
    gt_wTi1: Optional[Pose3] = None,
    gt_wTi2: Optional[Pose3] = None,
    gt_scene_mesh: Optional[Trimesh] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Checks the correspondences for epipolar distances and counts ones which are below the threshold.

    Args:
        keypoints_i1: keypoints in image i1.
        keypoints_i2: corr. keypoints in image i2.
        intrinsics_i1: intrinsics for i1.
        intrinsics_i2: intrinsics for i2.
        dist_threshold: max acceptable distance for a correct correspondence.
        gt_wTi1: ground truth pose of image i1.
        gt_wTi2: ground truth pose of image i2.
        gt_scene_mesh: ground truth triangular surface mesh of the scene in the world frame.

    Raises:
        ValueError: when the number of keypoints do not match.

    Returns:
        Boolean mask of which verified correspondences are classified as correct under Sampson error
            (using GT epipolar geometry).
        Reprojection error for every verified correspondence against GT geometry.
    """
    if corr_idxs_i1i2.size == 0:
        return None, None

    if gt_wTi1 is None or gt_wTi2 is None:
        return None, None

    # Compute ground truth correspondences.
    matched_keypoints_i1 = keypoints_i1.extract_indices(corr_idxs_i1i2[:, 0])
    matched_keypoints_i2 = keypoints_i2.extract_indices(corr_idxs_i1i2[:, 1])
    # Check to see if a GT mesh is provided.
    if gt_scene_mesh is not None:
        gt_camera_i1 = PinholeCameraCal3Bundler(gt_wTi1, intrinsics_i1)
        gt_camera_i2 = PinholeCameraCal3Bundler(gt_wTi2, intrinsics_i2)
        return mesh_inlier_correspondences(
            matched_keypoints_i1,
            matched_keypoints_i2,
            gt_camera_i1,
            gt_camera_i2,
            gt_scene_mesh,
            dist_threshold,
        )

    # If no mesh is provided, use squared Sampson error.
    gt_i2Ti1 = gt_wTi2.between(gt_wTi1)
    return epipolar_inlier_correspondences(
        matched_keypoints_i1,
        matched_keypoints_i2,
        intrinsics_i1,
        intrinsics_i2,
        gt_i2Ti1,
        dist_threshold,
    )


def epipolar_inlier_correspondences(
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    intrinsics_i1: Cal3Bundler,
    intrinsics_i2: Cal3Bundler,
    i2Ti1: Pose3,
    dist_threshold: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute inlier correspondences using epipolar geometry and the ground truth relative pose.

    Args:
        keypoints_i1: keypoints in image i1.
        keypoints_i2: corr. keypoints in image i2.
        intrinsics_i1: intrinsics for i1.
        intrinsics_i2: intrinsics for i2.
        i2Ti1: relative pose
        dist_threshold: max acceptable distance for a correct correspondence.

    Returns:
        is_inlier: (N, ) mask of inlier correspondences.
        distance_squared: squared sampson distance between corresponding keypoints.
    """
    i2Ei1 = EssentialMatrix(i2Ti1.rotation(), Unit3(i2Ti1.translation()))
    i2Fi1 = verification_utils.essential_to_fundamental_matrix(i2Ei1, intrinsics_i1, intrinsics_i2)
    distance_squared = verification_utils.compute_epipolar_distances_sq_sampson(
        keypoints_i1.coordinates, keypoints_i2.coordinates, i2Fi1
    )
    is_inlier = distance_squared < dist_threshold**2 if distance_squared is not None else None

    return is_inlier, distance_squared


def mesh_inlier_correspondences(
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    gt_camera_i1: PinholeCameraCal3Bundler,
    gt_camera_i2: PinholeCameraCal3Bundler,
    gt_scene_mesh: Trimesh,
    dist_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute inlier correspondences using the ground truth triangular surface mesh of the scene. First, rays are
    back-projected at each keypoint in the images and intersections between these rays and the ground truth mesh are
    recorded. Next, given a match, the mesh intersections corresponding to each keypoint are forward-projected into the
    other image and the reprojection error is computed to decide whether the match is an inlier.

    Args:
        keypoints_i1: N keypoints in image i1.
        keypoints_i2: N corresponding keypoints in image i2.
        gt_camera_i1: ground truth camera for image i1, i.e., wTi1 and intrinsics.
        gt_camera_i1: ground truth camera for image i2, i.e., wTi2 and intrinsics.
        gt_scene_mesh: ground truth triangular surface mesh of the scene in the world frame.
        dist_threshold: max acceptable reprojection error (in pixels) between image coordinates of ground truth landmark
            and keypoint.

    Returns:
        is_inlier: (N, ) mask of inlier correspondences.
        reproj_err: maximum error between forward-projected ground truth landmark and corresponding keypoints

    Raises:
        ValueError if the number of keypoints do not match.
    """
    if len(keypoints_i1) != len(keypoints_i2):
        raise ValueError("Keypoints must have same counts")
    n_corrs = len(keypoints_i1)
    is_inlier = np.zeros(n_corrs, dtype=bool)

    # Perform ray tracing to compute keypoint intersections.
    keypoint_ind_i1, intersections_i1 = compute_keypoint_intersections(keypoints_i1, gt_camera_i1, gt_scene_mesh)
    keypoint_ind_i2, intersections_i2 = compute_keypoint_intersections(keypoints_i2, gt_camera_i2, gt_scene_mesh)
    keypoint_ind, i1_idx, i2_idx = np.intersect1d(keypoint_ind_i1, keypoint_ind_i2, return_indices=True)

    # Forward project intersections into other image to compute error.
    reproj_err = np.array([np.nan] * len(keypoints_i1))
    for i in range(len(keypoint_ind)):
        uv_i1 = keypoints_i1.coordinates[keypoint_ind[i]]
        uv_i2 = keypoints_i2.coordinates[keypoint_ind[i]]
        uv_i2i1, success_flag_i1 = gt_camera_i1.projectSafe(intersections_i2[i2_idx[i]])
        uv_i1i2, success_flag_i2 = gt_camera_i2.projectSafe(intersections_i1[i1_idx[i]])
        if success_flag_i1 and success_flag_i2:
            err_i2i1 = np.linalg.norm(uv_i1 - uv_i2i1)
            err_i1i2 = np.linalg.norm(uv_i2 - uv_i1i2)
            is_inlier[keypoint_ind[i]] = max(err_i2i1, err_i1i2) < dist_threshold
            reproj_err[keypoint_ind[i]] = max(err_i2i1, err_i1i2)
        else:
            is_inlier[keypoint_ind[i]] = False
            reproj_err[keypoint_ind[i]] = np.nan

    return is_inlier, reproj_err


def compute_keypoint_intersections(
    keypoints: Keypoints, gt_camera: PinholeCameraCal3Bundler, gt_scene_mesh: Trimesh, verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes intersections between ground truth surface mesh and rays originating from image keypoints.

    Args:
        keypoints: N keypoints computed in image.
        gt_camera: ground truth camera.
        gt_scene_mesh: ground truth triangular surface mesh.

    Returns:
        keypoint_ind: (M,) array of keypoint indices whose corresponding ray intersected the ground truth mesh.
        intersections_locations: (M, 3), array of ray intersection locations.
    """
    num_kpts = len(keypoints)
    src = np.repeat(gt_camera.pose().translation().reshape((-1, 3)), num_kpts, axis=0)  # At_i1A
    drc = np.asarray([gt_camera.backproject(keypoints.coordinates[i], depth=1.0) - src[i, :] for i in range(num_kpts)])
    start_time = timeit.default_timer()
    intersections, keypoint_ind, _ = gt_scene_mesh.ray.intersects_location(src, drc, multiple_hits=False)
    if verbose:
        logger.debug("Case %d rays in %.6f seconds.", num_kpts, timeit.default_timer() - start_time)

    return keypoint_ind, intersections


def compute_rotation_angle_metric(wRi_list: List[Optional[Rot3]], gt_wRi_list: List[Optional[Pose3]]) -> GtsfmMetric:
    """Computes statistics for the angle between estimated and GT rotations.

    Assumes that the estimated and GT rotations have been aligned and do not
    have a gauge freedom.

    Args:
        wRi_list: List of estimated camera rotations.
        gt_wRi_list: List of ground truth camera rotations.

    Returns:
        A GtsfmMetric for the rotation angle errors, in degrees.
    """
    errors = []
    for (wRi, gt_wRi) in zip(wRi_list, gt_wRi_list):
        if wRi is not None and gt_wRi is not None:
            errors.append(comp_utils.compute_relative_rotation_angle(wRi, gt_wRi))
    return GtsfmMetric("rotation_angle_error_deg", errors)


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
    angles: List[Optional[float]] = []
    for (i1, i2) in i2Ui1_dict:
        i2Ui1 = i2Ui1_dict[(i1, i2)]
        angles.append(comp_utils.compute_translation_to_direction_angle(i2Ui1, wTi_list[i2], wTi_list[i1]))
    return GtsfmMetric("translation_angle_error_deg", np.array(angles, dtype=np.float))


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


def get_twoview_translation_directions(wTi_list: List[Optional[Pose3]]) -> Dict[Tuple[int, int], Optional[Unit3]]:
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
        if wTi_list[i1] is None or wTi_list[i2] is None:
            i2Ui1 = None
        else:
            i2Ti1 = wTi_list[i2].between(wTi_list[i1])
            i2Ui1 = Unit3(i2Ti1.translation())
        i2Ui1_dict[(i1, i2)] = i2Ui1
    return i2Ui1_dict


def get_precision_recall_from_errors(
    positive_errors: List[Optional[float]], negative_errors: List[Optional[float]], max_positive_error: float
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
    tp = np.sum(np.array(positive_errors, dtype=np.float32) <= max_positive_error)
    fp = np.sum(np.array(positive_errors, dtype=np.float32) > max_positive_error)
    fn = np.sum(np.array(negative_errors, dtype=np.float32) <= max_positive_error)

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


def save_metrics_as_json(metrics_groups: List[GtsfmMetricsGroup], output_dir: str) -> None:
    """Saves the input metrics groups as JSON files using the name of the group.

    Args:
        metrics_groups: List of GtsfmMetricsGroup to be saved.
        output_dir: Directory to save metrics to.
    """
    for metrics_group in metrics_groups:
        metrics_group.save_to_json(os.path.join(output_dir, metrics_group.name + ".json"))


def get_stats_for_sfmdata(gtsfm_data: GtsfmData, suffix: str) -> List[GtsfmMetric]:
    """Helper to get bundle adjustment metrics from a GtsfmData object with a suffix for metric names."""
    metrics = []
    metrics.append(GtsfmMetric(name="number_cameras", data=len(gtsfm_data.get_valid_camera_indices())))
    metrics.append(GtsfmMetric("number_tracks" + suffix, gtsfm_data.number_tracks()))
    metrics.append(
        GtsfmMetric(
            "3d_track_lengths" + suffix,
            gtsfm_data.get_track_lengths(),
            plot_type=GtsfmMetric.PlotType.HISTOGRAM,
        )
    )
    metrics.append(GtsfmMetric(f"reprojection_errors{suffix}_px", gtsfm_data.get_scene_reprojection_errors()))
    return metrics


def compute_percentage_change(x: float, y: float) -> float:
    """Return percentage in representing the regression or improvement of a value x, for new value y.

    Args:
        x: original value to compare against.
        y: new value.

    Returns:
        percentage change (may be positive or negative).
    """
    return (y - x) / (x + EPSILON) * 100
