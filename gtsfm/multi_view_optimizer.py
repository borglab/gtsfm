"""Optimizer which performs averaging and bundle adjustment on all images in the scene.

Authors: Ayush Baid, John Lambert
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from gtsam import Pose3, Rot3, Unit3

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.graph as graph_utils
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.bundle.global_ba import GlobalBundleAdjustment
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.data_association.data_assoc import DataAssociation
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.two_view_estimator import TwoViewEstimationReport
from gtsfm.view_graph_estimator.view_graph_estimator_base import ViewGraphEstimatorBase
from gtsfm.data_association.dsf_tracks_estimator import DsfTracksEstimator
from gtsfm.common.image import Image
from gtsfm.common.gtsfm_data import GtsfmData


class MultiViewOptimizer:
    def __init__(
        self,
        rot_avg_module: RotationAveragingBase,
        trans_avg_module: TranslationAveragingBase,
        data_association_module: DataAssociation,
        bundle_adjustment_module: GlobalBundleAdjustment,
        view_graph_estimator: Optional[ViewGraphEstimatorBase] = None,
    ) -> None:
        self.view_graph_estimator = view_graph_estimator
        self.rot_avg_module = rot_avg_module
        self.trans_avg_module = trans_avg_module
        self.data_association_module = data_association_module
        self.ba_optimizer = bundle_adjustment_module
        self._run_view_graph_estimator: bool = self.view_graph_estimator is not None

    def apply(
        self,
        images: List[Image],
        num_images: int,
        keypoints_list: List[Keypoints],
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        all_intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        two_view_reports_dict: Optional[Dict[Tuple[int, int], TwoViewEstimationReport]],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        gt_wTi_list: List[Optional[Pose3]],
    ) -> Tuple[GtsfmData, GtsfmData, Dict[Tuple[int, int], TwoViewEstimationReport], List[GtsfmMetricsGroup]]:
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

        if self._run_view_graph_estimator and self.view_graph_estimator is not None:
            (
                viewgraph_i2Ri1_dict,
                viewgraph_i2Ui1_dict,
                viewgraph_v_corr_idxs_dict,
                viewgraph_two_view_reports_dict,
                viewgraph_estimation_metrics,
            ) = self.view_graph_estimator.apply(
                i2Ri1_dict, i2Ui1_dict, all_intrinsics, v_corr_idxs_dict, keypoints_list, two_view_reports_dict
            )
        else:
            viewgraph_i2Ri1_dict = i2Ri1_dict
            viewgraph_i2Ui1_dict = i2Ui1_dict
            viewgraph_v_corr_idxs_dict = v_corr_idxs_dict
            viewgraph_two_view_reports_dict = two_view_reports_dict
            viewgraph_estimation_metrics = GtsfmMetricsGroup("view_graph_estimation_metrics", [])

        # prune the graph to a single connected component.
        pruned_i2Ri1_dict, pruned_i2Ui1_dict = graph_utils.prune_to_largest_connected_component(
            viewgraph_i2Ri1_dict, viewgraph_i2Ui1_dict, relative_pose_priors
        )

        wRi_list, rot_avg_metrics = self.rot_avg_module.apply(
            num_images, pruned_i2Ri1_dict, i1Ti2_priors=relative_pose_priors, wTi_gt=gt_wTi_list
        )
        tracks2d = get_2d_tracks(viewgraph_v_corr_idxs_dict, keypoints_list)

        wTi_list, ta_metrics = self.trans_avg_module.apply(
            num_images,
            pruned_i2Ui1_dict,
            wRi_list,
            tracks2d,
            all_intrinsics,
            absolute_pose_priors,
            relative_pose_priors,
            gt_wTi_list=gt_wTi_list,
        )
        cameras_initial = init_cameras(wTi_list, all_intrinsics)

        ba_input, data_assoc_metrics = self.data_association_module.apply(
            num_images,
            cameras_initial,
            tracks2d,
            cameras_gt,
            relative_pose_priors,
            images,
        )

        # align the sparse multi-view estimate before BA to the ground truth pose graph.
        ba_input = ba_input.align_via_Sim3_to_poses(gt_wTi_list)

        _, ba_output, _, ba_metrics = self.ba_optimizer.apply(
            ba_input, absolute_pose_priors, relative_pose_priors, cameras_gt
        )

        multiview_optimizer_metrics_graph: List[GtsfmMetricsGroup] = [
            viewgraph_estimation_metrics,
            rot_avg_metrics,
            ta_metrics,
            data_assoc_metrics,
            ba_metrics,
        ]

        return ba_input, ba_output, viewgraph_two_view_reports_dict, multiview_optimizer_metrics_graph


def init_cameras(
    wTi_list: List[Optional[Pose3]],
    intrinsics_list: List[gtsfm_types.CALIBRATION_TYPE],
) -> Dict[int, gtsfm_types.CAMERA_TYPE]:
    """Generate camera from valid rotations and unit-translations.

    Args:
        wTi_list: estimated global poses for cameras.
        intrinsics_list: intrinsics for cameras.

    Returns:
        Valid cameras.
    """
    cameras = {}

    camera_class = gtsfm_types.get_camera_class_for_calibration(intrinsics_list[0])
    for idx, (wTi) in enumerate(wTi_list):
        if wTi is not None:
            cameras[idx] = camera_class(wTi, intrinsics_list[idx])

    return cameras


def get_2d_tracks(
    corr_idxs_dict: Dict[Tuple[int, int], np.ndarray], keypoints_list: List[Keypoints]
) -> List[SfmTrack2d]:
    tracks_estimator = DsfTracksEstimator()
    return tracks_estimator.run(corr_idxs_dict, keypoints_list)
