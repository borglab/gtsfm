"""Optimizer which performs averaging and bundle adjustment on all images in the scene.

Authors: Ayush Baid, John Lambert
"""
from typing import Dict, List, Optional, Tuple

import dask
from dask.delayed import Delayed
from gtsam import Point3, Pose3, Rot3

import gtsfm.common.types as gtsfm_types
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.bundle.global_ba import GlobalBundleAdjustment
from gtsfm.common.pose_prior import PosePrior
from gtsfm.data_association.data_assoc import DataAssociation
from gtsfm.two_view_estimator import TwoViewEstimationReport
from gtsfm.view_graph_estimator.view_graph_estimator_base import ViewGraphEstimatorBase


class MultiViewOptimizer:
    def __init__(
        self,
        rot_avg_module: RotationAveragingBase,
        trans_avg_module: TranslationAveragingBase,
        data_association_module: DataAssociation,
        bundle_adjustment_module: GlobalBundleAdjustment,
        view_graph_estimator: ViewGraphEstimatorBase,
    ) -> None:
        self.view_graph_estimator = view_graph_estimator
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
        all_intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        two_view_reports_dict: Optional[Dict[Tuple[int, int], TwoViewEstimationReport]],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        gt_wTi_list: List[Optional[Pose3]],
    ) -> Tuple[Delayed, Delayed, Delayed, list]:
        """Creates a computation graph for multi-view optimization.

        Args:
            num_images: number of images in the scene.
            keypoints_graph: keypoints for images, each wrapped up as Delayed.
            i2Ri1_graph: relative rotations for image pairs, each value wrapped up as Delayed.
            i2Ui1_graph: relative unit-translations for image pairs, each value wrapped up as Delayed.
            v_corr_idxs_graph: indices of verified correspondences for image pairs, wrapped up as Delayed.
            all_intrinsics: intrinsics for images.
            absolute_pose_priors: priors on the camera poses (not delayed).
            relative_pose_priors: priors on the pose between camera pairs (not delayed)
            two_view_reports_dict: Dict of TwoViewEstimationReports after inlier support processor.
            cameras_gt: list of GT cameras (if they exist), ordered by camera index.
            gt_wTi_list: list of GT poses of the camera.

        Returns:
            The GtsfmData input to bundle adjustment, aligned to GT (if provided), wrapped up as Delayed.
            The final output GtsfmData, wrapped up as Delayed.
            Dict of TwoViewEstimationReports after view graph estimation.
            List of GtsfmMetricGroups from different modules, wrapped up as Delayed.
        """

        (
            pruned_i2Ri1_graph,
            pruned_i2Ui1_graph,
            viewgraph_v_corr_idxs_graph,
            viewgraph_two_view_reports_graph,
            viewgraph_estimation_metrics
        ) = self.view_graph_estimator.create_computation_graph(
            i2Ri1_graph,
            i2Ui1_graph,
            all_intrinsics,
            v_corr_idxs_graph,
            keypoints_graph,
            two_view_reports_dict,
            relative_pose_priors
        )

        delayed_wRi, rot_avg_metrics = self.rot_avg_module.create_computation_graph(
            num_images, pruned_i2Ri1_graph, i2Ti1_priors=relative_pose_priors, gt_wTi_list=gt_wTi_list
        )

        wti_graph, ta_metrics = self.trans_avg_module.create_computation_graph(
            num_images,
            pruned_i2Ui1_graph,
            delayed_wRi,
            absolute_pose_priors,
            relative_pose_priors,
            gt_wTi_list=gt_wTi_list,
        )
        init_cameras_graph = dask.delayed(init_cameras)(delayed_wRi, wti_graph, all_intrinsics)

        ba_input_graph, data_assoc_metrics_graph = self.data_association_module.create_computation_graph(
            num_images,
            init_cameras_graph,
            viewgraph_v_corr_idxs_graph,
            keypoints_graph,
            cameras_gt,
            relative_pose_priors,
            images_graph,
        )

        ba_result_graph, ba_metrics_graph = self.ba_optimizer.create_computation_graph(
            ba_input_graph, absolute_pose_priors, relative_pose_priors, cameras_gt
        )

        multiview_optimizer_metrics_graph = [
            viewgraph_estimation_metrics,
            rot_avg_metrics,
            ta_metrics,
            data_assoc_metrics_graph,
            ba_metrics_graph,
        ]

        # align the sparse multi-view estimate before BA to the ground truth pose graph.
        ba_input_graph = dask.delayed(ba_input_graph.align_via_Sim3_to_poses)(gt_wTi_list)

        return ba_input_graph, ba_result_graph, viewgraph_two_view_reports_graph, multiview_optimizer_metrics_graph


def init_cameras(
    wRi_list: List[Optional[Rot3]],
    wti_list: List[Optional[Point3]],
    intrinsics_list: List[gtsfm_types.CALIBRATION_TYPE],
) -> Dict[int, gtsfm_types.CAMERA_TYPE]:
    """Generate camera from valid rotations and unit-translations.

    Args:
        wRi_list: rotations for cameras.
        wti_list: translations for cameras.
        intrinsics_list: intrinsics for cameras.

    Returns:
        Valid cameras.
    """
    cameras = {}

    camera_class = gtsfm_types.get_camera_class_for_calibration(intrinsics_list[0])
    for idx, (wRi, wti) in enumerate(zip(wRi_list, wti_list)):
        if wRi is not None and wti is not None:
            cameras[idx] = camera_class(Pose3(wRi, wti), intrinsics_list[idx])

    return cameras
