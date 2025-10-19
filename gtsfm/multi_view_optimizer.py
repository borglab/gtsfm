"""Optimizer which performs averaging and bundle adjustment on all images in the scene.

Authors: Ayush Baid, John Lambert
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dask.delayed import Delayed, delayed
from gtsam import Pose3, Rot3, Unit3  # type: ignore

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.alignment as alignment_utils
import gtsfm.utils.graph as graph_utils
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.bundle.global_ba import GlobalBundleAdjustment
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.data_association.cpp_dsf_tracks_estimator import CppDsfTracksEstimator
from gtsfm.data_association.data_assoc import DataAssociation
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.products.one_view_data import OneViewData
from gtsfm.products.two_view_result import TwoViewResult
from gtsfm.products.visibility_graph import AnnotatedGraph
from gtsfm.view_graph_estimator.cycle_consistent_rotation_estimator import (
    CycleConsistentRotationViewGraphEstimator,
    EdgeErrorAggregationCriterion,
)
from gtsfm.view_graph_estimator.view_graph_estimator_base import ViewGraphEstimatorBase


class MultiViewOptimizer:
    @staticmethod
    def _extract_two_view_components(
        two_view_results: AnnotatedGraph[TwoViewResult],
    ) -> tuple[
        Dict[Tuple[int, int], Rot3],
        Dict[Tuple[int, int], Unit3],
        AnnotatedGraph[np.ndarray],
        AnnotatedGraph[TwoViewEstimationReport],
    ]:
        """Split TwoViewResult objects into the pieces needed by downstream modules."""

        i2Ri1_dict: Dict[Tuple[int, int], Rot3] = {}
        i2Ui1_dict: Dict[Tuple[int, int], Unit3] = {}
        v_corr_idxs_dict: AnnotatedGraph[np.ndarray] = {}
        two_view_reports: AnnotatedGraph[TwoViewEstimationReport] = {}

        for ij, result in two_view_results.items():
            i2Ri1_dict[ij] = result.i2Ri1
            i2Ui1_dict[ij] = result.i2Ui1
            v_corr_idxs_dict[ij] = result.v_corr_idxs
            two_view_reports[ij] = result.post_isp_report

        return i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, two_view_reports

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

        self.view_graph_estimator_v2 = CycleConsistentRotationViewGraphEstimator(
            edge_error_aggregation_criterion=EdgeErrorAggregationCriterion.MEDIAN_EDGE_ERROR
        )

    def __repr__(self) -> str:
        return f"""
        MultiviewOptimizer:
            ViewGraphEstimator: {self.view_graph_estimator}
            RotationAveraging: {self.rot_avg_module}
            TranslationAveraging: {self.trans_avg_module}
        """

    def create_computation_graph(
        self,
        keypoints_list: List[Keypoints],
        two_view_results: AnnotatedGraph[TwoViewResult],
        one_view_data_dict: Dict[int, OneViewData],
        image_delayed_map: Dict[int, Delayed],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        output_root: Optional[Path] = None,
    ) -> Tuple[Delayed, Delayed, Delayed, list]:
        """Creates a computation graph for multi-view optimization.

        Args:
            keypoints_list: Keypoints for images.
            two_view_results: valid two-view results for image pairs.
            one_view_data_dict: Per-view data entries keyed by image index.
            image_delayed_map: Delayed image fetch tasks keyed by image index.
            relative_pose_priors: Priors on the pose between camera pairs.
            output_root: Path where output should be saved.

        Returns:
            The GtsfmData input to bundle adjustment, aligned to GT (if provided), wrapped up as Delayed.
            The final output GtsfmData, wrapped up as Delayed.
            Dict of TwoViewEstimationReports after view graph estimation.
            List of GtsfmMetricGroups from different modules, wrapped up as Delayed.
        """

        (
            i2Ri1_dict,
            i2Ui1_dict,
            v_corr_idxs_dict,
            two_view_reports,
        ) = delayed(
            MultiViewOptimizer._extract_two_view_components, nout=4
        )(two_view_results)

        # Create debug directory.
        debug_output_dir = None
        if output_root:
            debug_output_dir = output_root / "debug"
            os.makedirs(debug_output_dir, exist_ok=True)

        num_images = len(one_view_data_dict)
        all_intrinsics = [one_view_data_dict[idx].intrinsics for idx in range(num_images)]
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
                two_view_reports,
                debug_output_dir,
            )

            # Second view graph estimator expects the same TwoViewResult format
            # Since ViewGraphEstimatorBase now uses the new signature, we pass two_view_results directly
            (
                viewgraph_i2Ri1_graph,
                viewgraph_i2Ui1_graph,
                viewgraph_v_corr_idxs_graph,
                viewgraph_two_view_reports_graph,
                viewgraph_estimation_metrics,
            ) = self.view_graph_estimator_v2.create_computation_graph(
                viewgraph_i2Ri1_graph,
                viewgraph_i2Ui1_graph,
                all_intrinsics,
                viewgraph_v_corr_idxs_graph,
                keypoints_list,
                viewgraph_two_view_reports_graph,
                debug_output_dir / "2",
            )
        else:
            viewgraph_i2Ri1_graph = i2Ri1_dict
            viewgraph_i2Ui1_graph = i2Ui1_dict
            viewgraph_v_corr_idxs_graph = v_corr_idxs_dict
            viewgraph_two_view_reports_graph = two_view_reports
            viewgraph_estimation_metrics = delayed(GtsfmMetricsGroup("view_graph_estimation_metrics", []))

        # Prune the graph to a single connected component.
        gt_wTi_list = [one_view_data_dict[idx].pose_gt for idx in range(num_images)]
        pruned_i2Ri1_graph, pruned_i2Ui1_graph = delayed(graph_utils.prune_to_largest_connected_component, nout=2)(
            viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph, relative_pose_priors
        )
        delayed_wRi, rot_avg_metrics = self.rot_avg_module.create_computation_graph(
            num_images,
            pruned_i2Ri1_graph,
            i1Ti2_priors=relative_pose_priors,
            gt_wTi_list=gt_wTi_list,
            v_corr_idxs=viewgraph_v_corr_idxs_graph,
        )
        tracks2d_graph = delayed(get_2d_tracks)(viewgraph_v_corr_idxs_graph, keypoints_list)

        absolute_pose_priors = [one_view_data_dict[idx].absolute_pose_prior for idx in range(num_images)]
        wTi_graph, ta_metrics, ta_inlier_idx_i1_i2 = self.trans_avg_module.create_computation_graph(
            num_images,
            pruned_i2Ui1_graph,
            delayed_wRi,
            tracks2d_graph,
            all_intrinsics,
            absolute_pose_priors,
            relative_pose_priors,
            gt_wTi_list=gt_wTi_list,
        )
        ta_v_corr_idxs_graph = delayed(filter_corr_by_idx)(viewgraph_v_corr_idxs_graph, ta_inlier_idx_i1_i2)
        ta_inlier_tracks_2d_graph = delayed(get_2d_tracks)(ta_v_corr_idxs_graph, keypoints_list)
        # TODO(akshay-krishnan): update pose priors also with the same inlier indices, right now these are unused.

        init_cameras_graph = delayed(init_cameras)(wTi_graph, all_intrinsics)

        cameras_gt = [one_view_data_dict[idx].camera_gt for idx in range(num_images)]
        images: List[Delayed] = [image_delayed_map[idx] for idx in range(num_images)]
        ba_input_graph, data_assoc_metrics_graph = self.data_association_module.create_computation_graph(
            num_images,
            init_cameras_graph,
            ta_inlier_tracks_2d_graph,
            cameras_gt,
            relative_pose_priors,
            images,
        )

        ba_result_graph, ba_metrics_graph = self.ba_optimizer.create_computation_graph(
            ba_input_graph,
            absolute_pose_priors,
            relative_pose_priors,
            cameras_gt,
            save_dir=str(output_root) if output_root else None,
        )

        multiview_optimizer_metrics_graph = [
            viewgraph_estimation_metrics,
            rot_avg_metrics,
            ta_metrics,
            data_assoc_metrics_graph,
            ba_metrics_graph,
        ]

        # Align the sparse multi-view estimate before BA to the ground truth pose graph.
        ba_input_graph = delayed(alignment_utils.align_gtsfm_data_via_Sim3_to_poses)(ba_input_graph, gt_wTi_list)

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


def get_2d_tracks(correspondences: AnnotatedGraph[np.ndarray], keypoints_list: List[Keypoints]) -> List[SfmTrack2d]:
    tracks_estimator = CppDsfTracksEstimator()
    return tracks_estimator.run(correspondences, keypoints_list)


def filter_corr_by_idx(correspondences: AnnotatedGraph[np.ndarray], idxs: List[Tuple[int, int]]):
    """Filter correspondences by indices.

    Args:
        correspondences: Correspondences as a dictionary.
        idxs: Indices to filter by.

    Returns:
        Filtered correspondences.
    """
    return {k: v for k, v in correspondences.items() if k in idxs}
