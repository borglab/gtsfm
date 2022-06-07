"""Optimizer which performs averaging and bundle adjustment on all images in the scene.

Authors: Ayush Baid, John Lambert
"""
from typing import Dict, List, Optional, Tuple, Mapping, Union

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Pose3, Rot3, Unit3

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.graph as graph_utils
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.pose_prior import PosePrior
from gtsfm.data_association.data_assoc import DataAssociation
from gtsfm.pose_slam.pose_slam import PoseSlam
from gtsfm.two_view_estimator import TwoViewEstimationReport
from gtsfm.view_graph_estimator.view_graph_estimator_base import ViewGraphEstimatorBase


class MultiViewOptimizer:
    def __init__(
        self,
        rot_avg_module: RotationAveragingBase,
        trans_avg_module: TranslationAveragingBase,
        data_association_module: DataAssociation,
        bundle_adjustment_module: BundleAdjustmentOptimizer,
        view_graph_estimator: Optional[ViewGraphEstimatorBase] = None,
        use_pose_slam_initialization: bool = False,
    ) -> None:
        self.view_graph_estimator = view_graph_estimator
        self.pose_slam_module = PoseSlam()
        self.rot_avg_module = rot_avg_module
        self.trans_avg_module = trans_avg_module
        self.data_association_module = data_association_module
        self.ba_optimizer = bundle_adjustment_module
        self._use_pose_slam_initialization = use_pose_slam_initialization

    def create_computation_graph(
        self,
        num_images: int,
        delayed_keypoints: List[Delayed],
        delayed_i2Ri1s: Mapping[Tuple[int, int], Union[Delayed, Optional[Rot3]]],
        delayed_i2Ui1s: Mapping[Tuple[int, int], Union[Delayed, Optional[Unit3]]],
        delayed_v_corr_idxs: Mapping[Tuple[int, int], Union[Delayed, Optional[np.ndarray]]],
        all_intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        two_view_reports_dict: Optional[Dict[Tuple[int, int], TwoViewEstimationReport]],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        gt_wTi_list: List[Optional[Pose3]],
        images_graph: Optional[List[Delayed]] = None,
    ) -> Tuple[Delayed, Delayed, Delayed, List[Delayed]]:
        """Creates a computation graph for multi-view optimization.

        Args:
            num_images: number of images in the scene.
            delayed_keypoints: keypoints for images, each wrapped up as Delayed.
            delayed_i2Ri1s: relative rotations for image pairs, each value wrapped up as Delayed.
            delayed_i2Ui1s: relative unit-translations for image pairs, each value wrapped up as Delayed.
            delayed_v_corr_idxs: indices of verified correspondences for image pairs, wrapped up as Delayed.
            all_intrinsics: intrinsics for images.
            absolute_pose_priors: priors on the camera poses.
            relative_pose_priors: priors on the pose between camera pairs.
            two_view_reports_dict:
            cameras_gt: list of GT cameras (if they exist), ordered by camera index.
            gt_wTi_list: list of GT poses of the camera.
            images_graph (optional): list of images. Defaults to None.

        Returns:
            The GtsfmData input to bundle adjustment, aligned to GT (if provided), wrapped up as Delayed.
            The final output GtsfmData, wrapped up as Delayed.
            List of GtsfmMetricGroups from different modules, wrapped up as Delayed.
        """
        metrics = []

        if self.view_graph_estimator is not None:
            (
                viewgraph_i2Ri1s,
                viewgraph_i2Ui1s,
                viewgraph_v_corr_idxs,
                viewgraph_twoview_reports,
                viewgraph_metrics,
            ) = self.view_graph_estimator.create_computation_graph(
                delayed_i2Ri1s,
                delayed_i2Ui1s,
                all_intrinsics,
                delayed_v_corr_idxs,
                delayed_keypoints,
                two_view_reports_dict,
            )
            metrics += [viewgraph_metrics]
        else:
            viewgraph_i2Ri1s = delayed_i2Ri1s
            viewgraph_i2Ui1s = delayed_i2Ui1s
            viewgraph_v_corr_idxs = delayed_v_corr_idxs
            viewgraph_twoview_reports = two_view_reports_dict

        pruned_i2Ri1s, pruned_i2Ui1s = dask.delayed(graph_utils.prune_to_largest_connected_component, nout=2)(
            viewgraph_i2Ri1s, viewgraph_i2Ui1s, relative_pose_priors
        )

        if self._use_pose_slam_initialization:
            delayed_poses, pose_slam_metrics = self.pose_slam_module.create_computation_graph(
                num_images, relative_pose_priors=relative_pose_priors, gt_wTi_list=gt_wTi_list
            )
            metrics.append(pose_slam_metrics)
        else:
            delayed_wRi, rot_avg_metrics = self.rot_avg_module.create_computation_graph(
                num_images, pruned_i2Ri1s, relative_pose_priors=relative_pose_priors, gt_wTi_list=gt_wTi_list
            )

            delayed_poses, ta_metrics = self.trans_avg_module.create_computation_graph(
                num_images,
                pruned_i2Ui1s,
                delayed_wRi,
                absolute_pose_priors,
                relative_pose_priors,
                gt_wTi_list=gt_wTi_list,
            )
            metrics += [rot_avg_metrics, ta_metrics]

        initialized_cameras = dask.delayed(init_cameras)(delayed_poses, all_intrinsics)

        ba_input, da_metrics = self.data_association_module.create_computation_graph(
            num_images,
            initialized_cameras,
            viewgraph_v_corr_idxs,
            delayed_keypoints,
            cameras_gt,
            relative_pose_priors,
            images_graph,
        )

        ba_output, ba_metrics = self.ba_optimizer.create_computation_graph(
            ba_input, absolute_pose_priors, relative_pose_priors, cameras_gt
        )

        metrics += [
            da_metrics,
            ba_metrics,
        ]

        # align the sparse multi-view estimate before BA to the ground truth pose graph.
        # TODO(Frank): Why would we do this here? But maybe we should simply fix align_via_Sim3_to_poses
        # ba_input_graph = dask.delayed(ba_input_graph.align_via_Sim3_to_poses)(gt_wTi_list)

        return ba_input, ba_output, viewgraph_twoview_reports, metrics


def init_cameras(
    wTi_list: List[Optional[Pose3]],
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
    camera_class = gtsfm_types.get_camera_class_for_calibration(intrinsics_list[0])
    return {idx: camera_class(wTi, intrinsics_list[idx]) for idx, wTi in enumerate(wTi_list) if wTi is not None}
