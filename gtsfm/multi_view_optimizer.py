"""Optimizer which performs averaging and bundle adjustment on all images in the scene.

Authors: Ayush Baid, John Lambert
"""
from typing import Dict, List, Optional, Tuple

import dask
from dask.delayed import Delayed
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3

import gtsfm.utils.graph as graph_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.data_association.data_assoc import DataAssociation
from gtsfm.evaluation.metrics import GtsfmMetricsGroup


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
        gt_cameras_graph: Optional[List[Delayed]] = None,
    ) -> Tuple[Delayed, Delayed, Delayed]:
        """Creates a computation graph for multi-view optimization.

        Args:
            num_images: number of images in the scene.
            keypoints_graph: keypoints for images, each wrapped up as Delayed.
            i2Ri1_graph: relative rotations for image pairs, each value wrapped up as Delayed.
            i2Ui1_graph: relative unit-translations for image pairs, each value wrapped up as Delayed.
            v_corr_idxs_graph: indices of verified correspondences for image pairs, wrapped up as Delayed.
            intrinsics_graph: intrinsics for images, wrapped up as Delayed.
            gt_cameras_graph: list of GT cameras (if they exist), ordered by camera index, wrapped up as Delayed.

        Returns:
            The GtsfmData input to bundle adjustment, aligned to GT (if provided), wrapped up as Delayed.
            The final output GtsfmData, wrapped up as Delayed.
            List of GtsfmMetricGroups from different modules, wrapped up as Delayed.
        """
        # prune the graph to a single connected component.
        pruned_graph = dask.delayed(graph_utils.prune_to_largest_connected_component)(i2Ri1_graph, i2Ui1_graph)

        pruned_i2Ri1_graph = pruned_graph[0]
        pruned_i2Ui1_graph = pruned_graph[1]

        gt_poses_graph = (
            [dask.delayed(lambda x: x.pose())(cam) for cam in gt_cameras_graph] if gt_cameras_graph else None
        )
        wRi_graph = self.rot_avg_module.create_computation_graph(num_images, pruned_i2Ri1_graph)
        wti_graph, ta_metrics = self.trans_avg_module.create_computation_graph(
            num_images, pruned_i2Ui1_graph, wRi_graph, gt_wTi_graph=gt_poses_graph
        )
        init_cameras_graph = dask.delayed(init_cameras)(wRi_graph, wti_graph, intrinsics_graph)

        ba_input_graph, data_assoc_metrics_graph = self.data_association_module.create_computation_graph(
            num_images, init_cameras_graph, v_corr_idxs_graph, keypoints_graph, images_graph, gt_cameras_graph
        )

        ba_result_graph, ba_metrics_graph = self.ba_optimizer.create_computation_graph(ba_input_graph, gt_cameras_graph)

        if gt_cameras_graph is None:
            return ba_input_graph, ba_result_graph, None

        rot_avg_metrics = dask.delayed(metrics_utils.compute_rotation_averaging_metrics)(
            wRi_graph, wti_graph, gt_poses_graph
        )
        averaging_metrics = dask.delayed(get_averaging_metrics)(rot_avg_metrics, ta_metrics)

        multiview_optimizer_metrics_graph = [averaging_metrics, data_assoc_metrics_graph, ba_metrics_graph]

        # align the sparse multi-view estimate before BA to the ground truth pose graph.
        ba_input_graph = dask.delayed(ba_input_graph.align_via_Sim3_to_poses)(gt_poses_graph)

        return ba_input_graph, ba_result_graph, multiview_optimizer_metrics_graph


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
