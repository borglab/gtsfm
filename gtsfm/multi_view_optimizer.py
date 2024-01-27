"""Optimizer which performs averaging and bundle adjustment on all images in the scene.

Authors: Ayush Baid, John Lambert
"""
from typing import Dict, List, Optional, Tuple

import dask
import numpy as np
import os
from dask.delayed import Delayed
from pathlib import Path
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
from gtsfm.view_graph_estimator.view_graph_estimator_base import ViewGraphEstimatorBase
from gtsfm.data_association.cpp_dsf_tracks_estimator import CppDsfTracksEstimator
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport


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

    def __repr__(self) -> str:
        return f"""
        MultiviewOptimizer: 
            ViewGraphEstimator: {self.view_graph_estimator}
            RotationAveraging: {self.rot_avg_module}
            TranslationAveraging: {self.trans_avg_module}
        """

    def create_computation_graph(
        self,
        images: List[Delayed],
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
        output_root: Optional[Path] = None,
    ) -> Tuple[Delayed, Delayed, Delayed, list]:
        """Creates a computation graph for multi-view optimization.

        Args:
            images: List of all images in the scene, as delayed.
            num_images: Number of images in the scene.
            keypoints_list: Keypoints for images.
            i2Ri1_dict: Relative rotations for image pairs.
            i2Ui1_dict: Relative unit-translations for image pairs.
            v_corr_idxs_dict: Indices of verified correspondences for image pairs.
            all_intrinsics: intrinsics for images.
            absolute_pose_priors: Priors on the camera poses.
            relative_pose_priors: Priors on the pose between camera pairs.
            two_view_reports_dict: Dict of TwoViewEstimationReports from the front-end.
            cameras_gt: List of GT cameras (if they exist), ordered by camera index.
            gt_wTi_list: List of GT poses of the camera.
            output_root: Path where output should be saved.

        Returns:
            The GtsfmData input to bundle adjustment, aligned to GT (if provided), wrapped up as Delayed.
            The final output GtsfmData, wrapped up as Delayed.
            Dict of TwoViewEstimationReports after view graph estimation.
            List of GtsfmMetricGroups from different modules, wrapped up as Delayed.
        """

        # Create debug directory.
        debug_output_dir = None
        if output_root:
            debug_output_dir = output_root / "debug"
            os.makedirs(debug_output_dir, exist_ok=True)

        if self._run_view_graph_estimator and self.view_graph_estimator is not None:
            (
                viewgraph_i2Ri1_graph,
                viewgraph_i2Ui1_graph,
                viewgraph_v_corr_idxs_graph,
                viewgraph_two_view_reports_graph,
                viewgraph_estimation_metrics,
            ) = self.view_graph_estimator.create_computation_graph(
                i2Ri1_dict,
                i2Ui1_dict,
                all_intrinsics,
                v_corr_idxs_dict,
                keypoints_list,
                two_view_reports_dict,
                debug_output_dir,
            )
        else:
            viewgraph_i2Ri1_graph = dask.delayed(i2Ri1_dict)
            viewgraph_i2Ui1_graph = dask.delayed(i2Ui1_dict)
            viewgraph_v_corr_idxs_graph = dask.delayed(v_corr_idxs_dict)
            viewgraph_two_view_reports_graph = dask.delayed(two_view_reports_dict)
            viewgraph_estimation_metrics = dask.delayed(GtsfmMetricsGroup("view_graph_estimation_metrics", []))

        # Prune the graph to a single connected component.
        pruned_i2Ri1_graph, pruned_i2Ui1_graph = dask.delayed(graph_utils.prune_to_largest_connected_component, nout=2)(
            viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph, relative_pose_priors
        )
        delayed_wRi, rot_avg_metrics = self.rot_avg_module.create_computation_graph(
            num_images, pruned_i2Ri1_graph, i1Ti2_priors=relative_pose_priors, gt_wTi_list=gt_wTi_list
        )
        tracks2d_graph = dask.delayed(get_2d_tracks)(viewgraph_v_corr_idxs_graph, keypoints_list)

        wTi_graph, ta_metrics, ta_inlier_idx_i1_i2 = self.trans_avg_module.create_computation_graph(
            num_images,
            pruned_i2Ui1_graph,
            delayed_wRi,
            viewgraph_v_corr_idxs_graph,
            keypoints_list,
            tracks2d_graph,
            all_intrinsics,
            absolute_pose_priors,
            relative_pose_priors,
            gt_wTi_list=gt_wTi_list,
        )
        ta_v_corr_idxs_graph = dask.delayed(filter_corr_by_idx)(viewgraph_v_corr_idxs_graph, ta_inlier_idx_i1_i2)
        ta_inlier_tracks_2d_graph = dask.delayed(get_2d_tracks)(ta_v_corr_idxs_graph, keypoints_list)
        # TODO(akshay-krishnan): update pose priors also with the same inlier indices, right now these are unused.

        init_cameras_graph = dask.delayed(init_cameras)(wTi_graph, all_intrinsics)

        ba_input_graph, data_assoc_metrics_graph = self.data_association_module.create_computation_graph(
            num_images,
            init_cameras_graph,
            ta_inlier_tracks_2d_graph,
            cameras_gt,
            relative_pose_priors,
            images,
        )
        ba_result_graph, ba_metrics_graph = self.ba_optimizer.create_computation_graph(
            ba_input_graph, absolute_pose_priors, relative_pose_priors, cameras_gt, save_dir=output_root
        )

        multiview_optimizer_metrics_graph = [
            viewgraph_estimation_metrics,
            rot_avg_metrics,
            ta_metrics,
            data_assoc_metrics_graph,
            ba_metrics_graph,
        ]

        # Align the sparse multi-view estimate before BA to the ground truth pose graph.
        ba_input_graph = dask.delayed(ba_input_graph.align_via_Sim3_to_poses)(gt_wTi_list)

        return ba_input_graph, ba_result_graph, viewgraph_two_view_reports_graph, multiview_optimizer_metrics_graph


def init_cameras(
    wTi_list: List[Optional[Pose3]],
    intrinsics_list: List[gtsfm_types.CALIBRATION_TYPE],
) -> Dict[int, gtsfm_types.CAMERA_TYPE]:
    """Generate camera from valid rotations and unit-translations.

    Args:
        wTi_list: Estimated global poses for cameras.
        intrinsics_list: Intrinsics for cameras.

    Returns:
        Valid cameras.
    """
    cameras = {}

    camera_class = gtsfm_types.get_camera_class_for_calibration(intrinsics_list[0])
    for idx, (wTi) in enumerate(wTi_list):
        if wTi is None:
            continue
        cameras[idx] = camera_class(wTi, intrinsics_list[idx])

    return cameras


def get_2d_tracks(
    corr_idxs_dict: Dict[Tuple[int, int], np.ndarray], keypoints_list: List[Keypoints]
) -> List[SfmTrack2d]:
    tracks_estimator = CppDsfTracksEstimator()
    return tracks_estimator.run(corr_idxs_dict, keypoints_list)


def filter_corr_by_idx(correspondences: Dict[Tuple[int, int], np.ndarray], idxs: List[Tuple[int, int]]):
    """Filter correspondences by indices.

    Args:
        correspondences: Correspondences as a dictionary.
        idxs: Indices to filter by.

    Returns:
        Filtered correspondences.
    """
    return {k: v for k, v in correspondences.items() if k in idxs}
