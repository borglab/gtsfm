"""Optimizer which performs averaging and bundle adjustment on all images in the scene.

Authors: Ayush Baid, John Lambert
"""
from typing import Any, Dict, List, Optional, Tuple

import dask
from dask.delayed import Delayed
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, Unit3

import gtsfm.averaging.rotation.cycle_consistency as cycle_consistency
import gtsfm.utils.graph as graph_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.two_view_estimator as two_view_estimator
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.data_association.data_assoc import DataAssociation
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.two_view_estimator import TwoViewEstimationReport

logger = logger_utils.get_logger()


# fmt: off
# Successive relaxation threshold pairs -- from strictest to loosest
# `inlier ratio` is the minimum allowed inlier ratio w.r.t. the estimated model
NUM_INLIERS_THRESHOLDS      =  [200, 175, 150, 125, 100, 75,  50,   25, 15] # noqa
MIN_INLIER_RATIOS_THRESHOLDS = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1] # noqa
# fmt: on

MEASUREMENT_TO_IMAGE_RATIO = 3


class MultiViewOptimizer:
    def __init__(
        self,
        rot_avg_module: RotationAveragingBase,
        trans_avg_module: TranslationAveragingBase,
        data_association_module: DataAssociation,
        bundle_adjustment_module: BundleAdjustmentOptimizer,
    ) -> None:
        self.rot_avg_module = rot_avg_module
        self.trans_avg_module = trans_avg_module
        self.data_association_module = data_association_module
        self.ba_optimizer = bundle_adjustment_module

    def create_computation_graph(
        self,
        images_graph: List[Delayed],
        num_images: int,
        keypoints_graph: List[Delayed],
        i2Ri1_graph: Dict[Tuple[int, int], Delayed],
        i2Ui1_graph: Dict[Tuple[int, int], Delayed],
        v_corr_idxs_graph: Dict[Tuple[int, int], Delayed],
        intrinsics_graph: List[Delayed],
        two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
        pose_angular_error_thresh: float,
        gt_poses_graph: Optional[List[Delayed]] = None,
    ) -> Tuple[Delayed, Delayed, Delayed]:
        """Creates a computation graph for multi-view optimization.

        Args:
            num_images: number of images in the scene.
            keypoints_graph: keypoints for images, each wrapped up as Delayed.
            i2Ri1_graph: relative rotations for image pairs, each value wrapped up as Delayed.
            i2Ui1_graph: relative unit-translations for image pairs, each value wrapped up as Delayed.
            v_corr_idxs_graph: indices of verified correspondences for image pairs, wrapped up as Delayed.
            intrinsics_graph: intrinsics for images, wrapped up as Delayed.
            two_view_reports_dict: front-end metrics for pairs of images.
            pose_angular_error_thresh:
            gt_poses_graph: list of GT camera poses, ordered by camera index (Pose3), wrapped up as Delayed

        Returns:
            The GtsfmData input to bundle adjustment, aligned to GT (if provided), wrapped up as Delayed.
            The final output GtsfmData, wrapped up as Delayed.
            List of GtsfmMetricGroups from different modules, wrapped up as Delayed.
        """

        i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph = dask.delayed(filter_edges_by_strictest_threshold, nout=3)(
            i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph, two_view_reports_dict, num_images
        )

        def _filter_dict_keys(dict: Dict[Any, Any], ref_dict: Dict[Any, Any]) -> Dict[Any, Any]:
            """Return a subset of a dictionary based on keys present in the reference dictionary."""
            valid_keys = list(ref_dict.keys())
            return {k: v for k, v in dict.items() if k in valid_keys}

        multiview_optimizer_metrics_graph = []
        if gt_poses_graph is not None:
            two_view_reports_dict_cycle_consistent = dask.delayed(_filter_dict_keys)(
                dict=two_view_reports_dict, ref_dict=i2Ri1_graph
            )
            multiview_optimizer_metrics_graph.append(
                dask.delayed(two_view_estimator.aggregate_frontend_metrics)(
                    two_view_reports_dict_cycle_consistent,
                    pose_angular_error_thresh,
                    metric_group_name="cycle_consistent_frontend_summary",
                )
            )

        # prune the graph to a single connected component.
        pruned_i2Ri1_graph, pruned_i2Ui1_graph = dask.delayed(graph_utils.prune_to_largest_connected_component, nout=2)(
            i2Ri1_graph, i2Ui1_graph
        )

        wRi_graph = self.rot_avg_module.create_computation_graph(num_images, pruned_i2Ri1_graph)
        wti_graph, ta_metrics = self.trans_avg_module.create_computation_graph(
            num_images, pruned_i2Ui1_graph, wRi_graph, gt_wTi_graph=gt_poses_graph
        )
        init_cameras_graph = dask.delayed(init_cameras)(wRi_graph, wti_graph, intrinsics_graph)

        ba_input_graph, data_assoc_metrics_graph = self.data_association_module.create_computation_graph(
            num_images, init_cameras_graph, v_corr_idxs_graph, keypoints_graph, images_graph
        )

        ba_result_graph, ba_metrics_graph = self.ba_optimizer.create_computation_graph(ba_input_graph, gt_poses_graph)

        if gt_poses_graph is None:
            return ba_input_graph, ba_result_graph, None

        rot_avg_metrics = dask.delayed(metrics_utils.compute_rotation_averaging_metrics)(
            wRi_graph, wti_graph, gt_poses_graph
        )
        averaging_metrics = dask.delayed(get_averaging_metrics)(rot_avg_metrics, ta_metrics)

        multiview_optimizer_metrics_graph.extend([averaging_metrics, data_assoc_metrics_graph, ba_metrics_graph])

        if gt_poses_graph is not None:
            # align the sparse multi-view estimate before BA to the ground truth pose graph.
            ba_input_graph = dask.delayed(ba_input_graph.align_via_Sim3_to_poses)(gt_poses_graph)

        return ba_input_graph, ba_result_graph, multiview_optimizer_metrics_graph


def filter_edges_by_strictest_threshold(
    i2Ri1_dict: Dict[Tuple[int, int], Delayed],
    i2Ui1_dict: Dict[Tuple[int, int], Delayed],
    v_corr_idxs_dict: Dict[Tuple[int, int], Delayed],
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    num_images: int,
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3]]:
    """Relax the strictness of front-end image pair acceptance thresholds until sufficient measurements are obtained.

    Empirically, we require at least (3 * number of images in the dataset) for # of backend measurements, to accept
    result. In other words, we use as a proxy "number of backend measurements coming out of cycle consistency" for
    "is the problem solvable". We only run the front-end computation once, however.

    Relaxation is necessary because no set of hyperparameters will generalize to all scenes, based on the width of
    baselines and # of total images. COLMAP does the same thing here:
        https://github.com/colmap/colmap/blob/dev/src/controllers/incremental_mapper.cc#L322

    We modify the following two thresholds:
    - min_num_inliers_acceptance: minimum number of inliers that must agree w/ estimated model, to use
        image pair.
    - min_allowed_inlier_ratio_est_model: minimum allowed inlier ratio w.r.t. the estimated model to accept
        the verification result and use the image pair, i.e. the lowest allowed ratio of
        #final RANSAC inliers/ #putatives. A lower fraction indicates less agreement among the result.

    Args:
        i2Ri1_dict: relative rotations for image pairs.
        i2Ui1_dict: relative unit-translations for image pairs.
        v_corr_idxs_graph: indices of verified correspondences for image pairs.
        two_view_reports_dict: front-end metrics for pairs of images.
        num_images: number of images in the scene.

    Returns:
        i2Ri1_dict_cc: relative rotations for cycle-consistent, high-confidence image pairs.
        i2Ui1_dict_cc: relative unit-translations for cycle-consistent, high-confidence image pairs.
        v_corr_idxs_dict_cc: indices of verified correspondences for cycle-consistent, high-confidence image pairs.
    """
    # try to relax the problem repeatedly
    for (min_num_inliers_acceptance, min_allowed_inlier_ratio_est_model) in zip(
        NUM_INLIERS_THRESHOLDS, MIN_INLIER_RATIOS_THRESHOLDS
    ):
        logger.info("New #inliers threshold:  %d inliers", min_num_inliers_acceptance)
        logger.info("New min. inlier ratio threshold: %.1f inlier ratio", min_allowed_inlier_ratio_est_model)

        high_confidence_edges = []

        # loop through all the 2-view reports. keep the ones where
        for (i1, i2), report in two_view_reports_dict.items():

            sufficient_inliers = report.num_inliers_est_model >= min_num_inliers_acceptance
            sufficient_inlier_ratio = report.inlier_ratio_est_model >= min_allowed_inlier_ratio_est_model

            sufficient_support = sufficient_inliers and sufficient_inlier_ratio
            if sufficient_support:
                high_confidence_edges.append((i1, i2))

        def _filter_dict_keys(dict: Dict[Any, Any], valid_keys: List[Tuple[int, int]]) -> Dict[Any, Any]:
            """Return a subset of a dictionary based on a specified list of valid keys."""
            return {k: v for k, v in dict.items() if k in valid_keys}

        # filter to this subset of edges
        i2Ri1_dict_conf = _filter_dict_keys(dict=i2Ri1_dict, valid_keys=high_confidence_edges)
        i2Ui1_dict_conf = _filter_dict_keys(dict=i2Ui1_dict, valid_keys=high_confidence_edges)
        v_corr_idxs_dict_conf = _filter_dict_keys(dict=v_corr_idxs_dict, valid_keys=high_confidence_edges)

        # ensure cycle consistency in triplets
        i2Ri1_dict_cc, i2Ui1_dict_cc, v_corr_idxs_dict_cc = cycle_consistency.filter_to_cycle_consistent_edges(
            i2Ri1_dict_conf, i2Ui1_dict_conf, v_corr_idxs_dict_conf, two_view_reports_dict
        )

        # check for success
        num_backend_input_pairs = len(i2Ri1_dict_cc)
        num_required_backend_input_pairs = MEASUREMENT_TO_IMAGE_RATIO * num_images
        if num_backend_input_pairs < num_required_backend_input_pairs:
            logger.info("Too few measurements at this threshold, will try relaxing the problem...")
            logger.info(
                "Found only %d num_backend_input_pairs, needed %d",
                num_backend_input_pairs,
                num_required_backend_input_pairs,
            )
            logger.exception("Computation was unsuccessful, will try relaxing the problem ...")
        else:
            logger.info(
                "GTSFM Succeeded, with %d num_backend_input_pairs, needed %d",
                num_backend_input_pairs,
                num_required_backend_input_pairs,
            )
            return i2Ri1_dict_cc, i2Ui1_dict_cc, v_corr_idxs_dict_cc

    logger.error("No problem relaxation yielded sufficient number of edges for back-end optimization. Aborting...")
    exit()


def init_cameras(
    wRi_list: List[Optional[Rot3]],
    wti_list: List[Optional[Point3]],
    intrinsics_list: List[Cal3Bundler],
) -> Dict[int, PinholeCameraCal3Bundler]:
    """Generate camera from valid rotations and unit-translations.

    Args:
        wRi_list: rotations for cameras.
        wti_list: translations for cameras.
        intrinsics_list: intrinsics for cameras.

    Returns:
        Valid cameras.
    """
    cameras = {}

    for idx, (wRi, wti) in enumerate(zip(wRi_list, wti_list)):
        if wRi is not None and wti is not None:
            cameras[idx] = PinholeCameraCal3Bundler(Pose3(wRi, wti), intrinsics_list[idx])

    return cameras


def get_averaging_metrics(
    rot_avg_metrics: GtsfmMetricsGroup, trans_avg_metrics: GtsfmMetricsGroup
) -> GtsfmMetricsGroup:
    """Helper to combine rotation and translation averaging metrics groups into a single averaging metrics group.

    Args:
        rot_avg_metrics: Rotation averaging metrics group.
        trans_avg_metrics: Translation averaging metrics group.

    Returns:
        An averaging metrics group with both rotation and translation averaging metrics.
    """
    return GtsfmMetricsGroup("averaging_metrics", rot_avg_metrics.metrics + trans_avg_metrics.metrics)
