"""Optimizer which performs averaging and bundle adjustment on all images in the scene.

Authors: Ayush Baid, John Lambert
"""
from typing import Dict, List, Optional, Tuple

import dask
from dask.delayed import Delayed
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, Unit3

import gtsfm.utils.graph as graph_utils
import gtsfm.utils.geometry_comparisons as geom_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.data_association.data_assoc import DataAssociation
from gtsfm.evaluation.metrics import GtsfmMetricsGroup

POST_ROTATION_AVERAGING_OUTLIER_REMOVAL_ANGULAR_THRESHOLD_DEGREES = 10

logger = logger_utils.get_logger()


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
            gt_poses_graph: list of GT camera poses, ordered by camera index (Pose3), wrapped up as Delayed

        Returns:
            The GtsfmData input to bundle adjustment, aligned to GT (if provided), wrapped up as Delayed.
            The final output GtsfmData, wrapped up as Delayed.
            List of GtsfmMetricGroups from different modules, wrapped up as Delayed.
        """
        # prune the graph to a single connected component.
        pruned_graph = dask.delayed(graph_utils.prune_to_largest_connected_component)(i2Ri1_graph, i2Ui1_graph)

        pruned_i2Ri1_graph = pruned_graph[0]

        wRi_graph = self.rot_avg_module.create_computation_graph(num_images, pruned_i2Ri1_graph)
        filtered_graph = dask.delayed(filter_inconsistent_pairwise_rotations)(wRi_graph, i2Ri1_graph, i2Ui1_graph)
        filtered_i2Ui1_graph = filtered_graph[1]
        wti_graph, ta_metrics = self.trans_avg_module.create_computation_graph(
            num_images, filtered_i2Ui1_graph, wRi_graph, gt_wTi_graph=gt_poses_graph
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

        multiview_optimizer_metrics_graph = [averaging_metrics, data_assoc_metrics_graph, ba_metrics_graph]

        if gt_poses_graph is not None:
            # align the sparse multi-view estimate before BA to the ground truth pose graph.
            ba_input_graph = dask.delayed(ba_input_graph.align_via_Sim3_to_poses)(gt_poses_graph)

        return ba_input_graph, ba_result_graph, multiview_optimizer_metrics_graph


def filter_inconsistent_pairwise_rotations(
    wRi_list: List[Optional[Rot3]],
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3]]:
    """[summary]

    Args:
        wRi_list (List[Optional[Rot3]]): [description]
        i2Ri1_dict (Dict[Tuple[int, int], Optional[Rot3]]): [description]
        i2Ui1_dict (Dict[Tuple[int, int], Optional[Unit3]]): [description]

    Returns:
        Tuple[Dict[Tuple[int, int], Optional[Rot3]], Dict[Tuple[int, int], Optional[Unit3]]]: [description]
    """

    # keep the pairwise rotations which agree with the global rotations. mirror the keys in i2Ui1s.
    filtered_i2Ri1_dict: Dict[Tuple[int, int], Rot3] = dict()
    filtered_i2Ui1_dict: Dict[Tuple[int, int], Unit3] = dict()
    for i1i2, i2Ri1 in i2Ri1_dict.items():
        if i2Ri1 is None:
            continue
        i1, i2 = i1i2
        wRi1 = wRi_list[i1]
        wRi2 = wRi_list[i2]
        if wRi1 is None or wRi2 is None:
            continue
        i2Ri1_from_global = wRi2.between(wRi1)

        angular_error = geom_utils.compute_relative_rotation_angle(i2Ri1, i2Ri1_from_global)
        if angular_error < POST_ROTATION_AVERAGING_OUTLIER_REMOVAL_ANGULAR_THRESHOLD_DEGREES:
            filtered_i2Ri1_dict[(i1, i2)] = i2Ri1
            filtered_i2Ui1_dict[(i1, i2)] = i2Ui1_dict[(i1, i2)]

    input_len = len(i2Ri1_dict)
    output_len = len(filtered_i2Ri1_dict)
    logger.info("Dropped %d/%d unit-translations post rotation averaging", input_len - output_len, input_len)

    return filtered_i2Ri1_dict, filtered_i2Ui1_dict


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
