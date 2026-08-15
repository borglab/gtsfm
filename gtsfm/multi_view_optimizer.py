"""Optimizer which performs averaging and bundle adjustment on all images in the scene.

Authors: Ayush Baid, John Lambert
"""

import dataclasses
import logging
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np
from dask.delayed import Delayed, delayed
from dask.distributed import Future
from gtsam import Pose3, Rot3, Unit3  # type: ignore

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.graph as graph_utils
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.bundle.global_ba import GlobalBundleAdjustment
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.data_association.cpp_dsf_tracks_estimator import CppDsfTracksEstimator
from gtsfm.data_association.data_assoc import DataAssociation
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.global_positioner.global_positioner import GlobalPositioner
from gtsfm.products.one_view_data import OneViewData
from gtsfm.products.two_view_result import TwoViewResult
from gtsfm.products.visibility_graph import AnnotatedGraph
from gtsfm.utils import verification as verification_utils
from gtsfm.view_graph_estimator import view_graph_calibration
from gtsfm.view_graph_estimator.view_graph_estimator_base import ViewGraphEstimatorBase

logger = logging.getLogger(__name__)


class MultiViewOptimizer:
    @staticmethod
    def _extract_two_view_components(
        two_view_results: AnnotatedGraph[TwoViewResult],
    ) -> tuple[
        Dict[Tuple[int, int], Rot3],
        Dict[Tuple[int, int], Unit3],
        AnnotatedGraph[np.ndarray],
        AnnotatedGraph[TwoViewEstimationReport],
        AnnotatedGraph[PosePrior],
        Dict[Tuple[int, int], np.ndarray],
        Dict[Tuple[int, int], int],
    ]:
        """Split TwoViewResult objects into the pieces needed by downstream modules."""

        i2Ri1_dict: Dict[Tuple[int, int], Rot3] = {}
        i2Ui1_dict: Dict[Tuple[int, int], Unit3] = {}
        v_corr_idxs_dict: AnnotatedGraph[np.ndarray] = {}
        two_view_reports: AnnotatedGraph[TwoViewEstimationReport] = {}
        relative_pose_priors: AnnotatedGraph[PosePrior] = {}
        # Focal-independent F + two-view config per edge (consumed by the closed-form Fetzer calibration).
        i2Fi1_dict: Dict[Tuple[int, int], np.ndarray] = {}
        i2_config_dict: Dict[Tuple[int, int], int] = {}

        for ij, result in two_view_results.items():
            i2Ri1_dict[ij] = result.i2Ri1
            i2Ui1_dict[ij] = result.i2Ui1
            v_corr_idxs_dict[ij] = result.v_corr_idxs
            two_view_reports[ij] = result.post_isp_report
            if result.relative_pose_prior is not None:
                relative_pose_priors[ij] = result.relative_pose_prior
            if result.i2Fi1 is not None:
                i2Fi1_dict[ij] = result.i2Fi1
            if result.config is not None:
                i2_config_dict[ij] = result.config

        return (
            i2Ri1_dict,
            i2Ui1_dict,
            v_corr_idxs_dict,
            two_view_reports,
            relative_pose_priors,
            i2Fi1_dict,
            i2_config_dict,
        )

    def __init__(
        self,
        rot_avg_module: RotationAveragingBase,
        trans_avg_module: TranslationAveragingBase,
        data_association_module: DataAssociation,
        bundle_adjustment_module: GlobalBundleAdjustment,
        view_graph_estimator: Optional[ViewGraphEstimatorBase] = None,
        global_positioner: Optional[GlobalPositioner] = None,
        run_view_graph_calibration: bool = False,
    ) -> None:
        self.view_graph_estimator = view_graph_estimator
        self.rot_avg_module = rot_avg_module
        self.trans_avg_module = trans_avg_module
        self.data_association_module = data_association_module
        self.ba_optimizer = bundle_adjustment_module
        self.global_positioner = global_positioner
        self._run_view_graph_estimator: bool = self.view_graph_estimator is not None
        self._run_view_graph_calibration = run_view_graph_calibration

    def __repr__(self) -> str:
        return f"""
        MultiviewOptimizer:
            ViewGraphEstimator: {self.view_graph_estimator}
            RotationAveraging: {self.rot_avg_module}
            TranslationAveraging: {self.trans_avg_module}
        """

    def create_computation_graph(
        self,
        keypoints_graph: Delayed,
        two_view_results_graph: Delayed,
        one_view_data_dict: Dict[int, OneViewData],
        image_future_map: Mapping[int, Future],
        output_root: Optional[Path] = None,
    ) -> Tuple[Delayed, Delayed, Delayed, list]:
        """Creates a computation graph for multi-view optimization.

        Args:
            keypoints_graph: Delayed task producing padded keypoints for images.
            two_view_results_graph: Delayed task producing valid two-view results for image pairs.
            one_view_data_dict: Per-view data entries keyed by image index.
            image_future_map: Mapping of image index to cached futures for `get_image`.
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
            pose_priors_graph,
            i2Fi1_dict,
            i2_config_dict,
        ) = delayed(MultiViewOptimizer._extract_two_view_components, nout=7)(two_view_results_graph)

        # Create debug directory.
        debug_output_dir = None
        if output_root:
            debug_output_dir = output_root / "debug"
            os.makedirs(debug_output_dir, exist_ok=True)

        num_images = len(one_view_data_dict)
        all_intrinsics = [one_view_data_dict[idx].intrinsics for idx in one_view_data_dict.keys()]
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
                keypoints_graph,
                two_view_reports,
                debug_output_dir,
            )

            # NOTE: the second cycle-consistency pass (view_graph_estimator_v2) is intentionally
            # NOT run — on Brussels it over-prunes the view graph (9722 -> ~4565 edges), starving
            # the Fetzer focal calibration and global positioning. A single pass matches the tuned run.
        else:
            viewgraph_i2Ri1_graph = i2Ri1_dict
            viewgraph_i2Ui1_graph = i2Ui1_dict
            viewgraph_v_corr_idxs_graph = v_corr_idxs_dict
            viewgraph_two_view_reports_graph = two_view_reports
            viewgraph_estimation_metrics = delayed(GtsfmMetricsGroup("view_graph_estimation_metrics", []))

        # Re-score inliers per two-view config (GLOMAP ScoreError dispatch): planar/panoramic pairs
        # keep their homography inliers; uncalibrated pairs are scored against the stored focal-
        # independent F. Then drop weak pairs (GLOMAP FilterInlierNum=30 / FilterInlierRatio=0.25).
        viewgraph_v_corr_idxs_graph, viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph = delayed(
            rescore_inliers_fundamental, nout=3
        )(
            viewgraph_i2Ri1_graph,
            viewgraph_i2Ui1_graph,
            viewgraph_v_corr_idxs_graph,
            keypoints_graph,
            all_intrinsics,
            min_inlier_count=30,
            min_inlier_ratio=0.25,
            config_dict=i2_config_dict,
            i2Fi1_dict=i2Fi1_dict,
        )

        # View graph calibration: refine focal lengths from F-matrices
        if self._run_view_graph_calibration:
            all_intrinsics, edges_to_remove = delayed(view_graph_calibration.calibrate_view_graph, nout=2)(
                viewgraph_v_corr_idxs_graph,
                keypoints_graph,
                all_intrinsics,
                num_images,
                i2Fi1_dict=i2Fi1_dict,
                config_dict=i2_config_dict,
            )
            # Remove edges with high calibration error
            viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph, viewgraph_v_corr_idxs_graph = delayed(_filter_edges, nout=3)(
                viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph, viewgraph_v_corr_idxs_graph, edges_to_remove
            )
            # Re-estimate relative poses with refined intrinsics
            viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph = delayed(reestimate_relative_poses, nout=2)(
                viewgraph_i2Ri1_graph,
                viewgraph_i2Ui1_graph,
                viewgraph_v_corr_idxs_graph,
                keypoints_graph,
                all_intrinsics,
            )
            viewgraph_two_view_reports_graph = delayed(_sync_two_view_reports_after_calibration)(
                viewgraph_two_view_reports_graph,
                viewgraph_i2Ri1_graph,
                viewgraph_i2Ui1_graph,
                viewgraph_v_corr_idxs_graph,
            )

        # Prune the graph to a single connected component.
        gt_wTi = {k: val.pose_gt for k, val in one_view_data_dict.items()}
        gt_wTi_list = [gt_wTi[i] for i in sorted(list(gt_wTi.keys()))]
        pruned_i2Ri1_graph, pruned_i2Ui1_graph = delayed(graph_utils.prune_to_largest_connected_component, nout=2)(
            viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph, pose_priors_graph
        )
        delayed_wRi, rot_avg_metrics = self.rot_avg_module.create_computation_graph(
            num_images,
            pruned_i2Ri1_graph,
            i1Ti2_priors=pose_priors_graph,
            gt_wTi_list=gt_wTi_list,
            v_corr_idxs=viewgraph_v_corr_idxs_graph,
        )
        tracks2d_graph = delayed(get_2d_tracks)(viewgraph_v_corr_idxs_graph, keypoints_graph)

        absolute_pose_priors = [one_view_data_dict[idx].absolute_pose_prior for idx in range(num_images)]
        cameras_gt = [one_view_data_dict[idx].camera_gt for idx in sorted(list(one_view_data_dict.keys()))]

        if self.global_positioner is not None:
            # Path B: Global positioner replaces trans_avg + data_assoc.
            ba_input_graph, gp_metrics = delayed(self.global_positioner.run, nout=2)(
                num_images,
                delayed_wRi,
                tracks2d_graph,
                all_intrinsics,
                output_root=output_root,
            )
            ta_metrics = gp_metrics
            data_assoc_metrics_graph = delayed(GtsfmMetricsGroup)("data_association_metrics", [])
        else:
            # Path A: Existing pipeline — trans_avg + data_assoc.
            wTi_graph, ta_metrics, ta_inlier_idx_i1_i2 = self.trans_avg_module.create_computation_graph(
                num_images,
                pruned_i2Ui1_graph,
                delayed_wRi,
                tracks2d_graph,
                all_intrinsics,
                absolute_pose_priors,
                pose_priors_graph,
                gt_wTi_list=gt_wTi_list,
            )
            ta_v_corr_idxs_graph = delayed(filter_corr_by_idx)(viewgraph_v_corr_idxs_graph, ta_inlier_idx_i1_i2)
            ta_inlier_tracks_2d_graph = delayed(get_2d_tracks)(ta_v_corr_idxs_graph, keypoints_graph)
            # TODO(akshay-krishnan): update pose priors also with the same inlier indices, right now these are unused.

            init_cameras_graph = delayed(init_cameras)(wTi_graph, all_intrinsics)

            images: List[Future] = [image_future_map[idx] for idx in sorted(list(one_view_data_dict.keys()))]
            ba_input_graph, data_assoc_metrics_graph = self.data_association_module.create_computation_graph(
                num_images,
                init_cameras_graph,
                ta_inlier_tracks_2d_graph,
                cameras_gt,
                pose_priors_graph,
                images,
            )

        ba_result_graph, ba_metrics_graph = self.ba_optimizer.create_computation_graph(
            ba_input_graph,
            absolute_pose_priors,
            pose_priors_graph,
            cameras_gt,
            save_dir=str(output_root) if output_root else None,
            tracks_2d=tracks2d_graph,
        )

        multiview_optimizer_metrics_graph = [
            viewgraph_estimation_metrics,
            rot_avg_metrics,
            ta_metrics,
            data_assoc_metrics_graph,
            ba_metrics_graph,
        ]

        # This conversion is OK since the GT poses are expected to be complete.
        gt_wTi_dict: dict[int, Pose3] = {
            i: gt_wTi_list[i] for i in range(len(gt_wTi_list)) if gt_wTi_list[i] is not None
        }

        # Align the sparse multi-view estimate before BA to the ground truth pose graph.
        ba_input_graph = delayed(GtsfmData.align_via_sim3_and_transform)(ba_input_graph, gt_wTi_dict)

        return ba_input_graph, ba_result_graph, viewgraph_two_view_reports_graph, multiview_optimizer_metrics_graph


# pycolmap two-view ConfigurationType values (see gric_verifier.ConfigurationType).
_PLANAR_CONFIGS = frozenset({4, 5, 6})  # PLANAR / PANORAMIC / PLANAR_OR_PANORAMIC
_UNCALIBRATED_CONFIG = 3


def rescore_inliers_fundamental(
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    v_corr_idxs_dict: AnnotatedGraph[np.ndarray],
    keypoints_list: List[Keypoints],
    intrinsics: List[gtsfm_types.CALIBRATION_TYPE],
    max_sampson_error_px: float = 4.0,
    min_inlier_count: int = 0,
    min_inlier_ratio: float = 0.0,
    config_dict: Optional[Dict[Tuple[int, int], int]] = None,
    i2Fi1_dict: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
) -> Tuple[AnnotatedGraph[np.ndarray], Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3]]:
    """Re-filter each edge's correspondences against a fundamental matrix, then drop weak pairs.

    Mirrors GLOMAP's ScoreError() dispatch — the F we score against depends on the two-view config:
      - PLANAR / PANORAMIC (config 4-6): keep the verifier's inliers as-is. They are homography
        inliers; a fundamental matrix can't describe a plane and would wrongly gut these edges
        (which still matter for rotation averaging / GP — only their F is gated out of Fetzer).
      - UNCALIBRATED (config 3): score against the stored, focal-independent F (`i2Fi1_dict`).
      - CALIBRATED (config 2) or no config: score against an F rebuilt from the relative pose.

    A match survives if its Sampson distance to F is small AND its epipolar orientation ("signum")
    agrees with the dominant orientation of that edge's inliers — an oriented-epipolar / cheirality
    test (GC-RANSAC, GLOMAP). Pairs left with fewer than `min_inlier_count` matches, or below
    `min_inlier_ratio` of their original matches, are dropped (GLOMAP FilterInlierNum / Ratio).

    Returns the rescored correspondences and the rotation/translation dicts with dropped pairs removed.
    """
    from gtsam import EssentialMatrix

    max_sampson_sq = max_sampson_error_px ** 2
    rescored: Dict[Tuple[int, int], np.ndarray] = {}
    total_before = total_after = 0

    for (i1, i2), v_corr_idxs in v_corr_idxs_dict.items():
        i2Ri1, i2Ui1 = i2Ri1_dict.get((i1, i2)), i2Ui1_dict.get((i1, i2))
        # Nothing to score (no matches or no relative pose): keep as-is, don't count it.
        if v_corr_idxs.shape[0] == 0 or i2Ri1 is None or i2Ui1 is None:
            rescored[(i1, i2)] = v_corr_idxs
            continue
        total_before += len(v_corr_idxs)

        # Planar/panoramic: matches are homography inliers a fundamental matrix would wrongly
        # reject — keep them as-is (their F is excluded from Fetzer separately).
        config = config_dict.get((i1, i2)) if config_dict else None
        if config in _PLANAR_CONFIGS:
            rescored[(i1, i2)] = v_corr_idxs
            total_after += len(v_corr_idxs)
            continue

        # UNCALIBRATED uses the stored focal-independent F; otherwise rebuild it from the relative
        # pose: F = K2^-T [t]_x R K1^-1.
        F = i2Fi1_dict.get((i1, i2)) if (config == _UNCALIBRATED_CONFIG and i2Fi1_dict) else None
        if F is None:
            K1, K2 = intrinsics[i1].K(), intrinsics[i2].K()
            F = np.linalg.inv(K2).T @ EssentialMatrix(i2Ri1, i2Ui1).matrix() @ np.linalg.inv(K1)
        F = np.asarray(F)

        # Epipole (right null space of F), used by the orientation-signum test below.
        epipole = np.cross(F[0], F[2])
        if np.linalg.norm(epipole) < 1e-12:
            epipole = np.cross(F[1], F[2])

        pts1 = keypoints_list[i1].coordinates[v_corr_idxs[:, 0]]
        pts2 = keypoints_list[i2].coordinates[v_corr_idxs[:, 1]]
        pts1_h = np.column_stack([pts1, np.ones(len(pts1))])
        pts2_h = np.column_stack([pts2, np.ones(len(pts2))])

        # Sampson distance of every match to F's epipolar geometry (vectorized over all matches).
        Fx1 = pts1_h @ F.T   # F @ x1, per row
        Ftx2 = pts2_h @ F    # F^T @ x2, per row
        x2Fx1 = np.einsum("ij,ij->i", pts2_h, Fx1)
        denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-12
        sampson_inlier = (x2Fx1 ** 2 / denom) < max_sampson_sq

        # Oriented-epipolar (cheirality) test: keep matches whose orientation signum matches the
        # dominant signum among the Sampson inliers.
        signum = (F[0, 0] * pts2[:, 0] + F[1, 0] * pts2[:, 1] + F[2, 0]) * (epipole[1] - epipole[2] * pts1[:, 1])
        n_pos = int(((signum > 0) & sampson_inlier).sum())
        dominant_positive = n_pos > int(sampson_inlier.sum()) - n_pos
        keep_mask = sampson_inlier & ((signum > 0) == dominant_positive)

        rescored[(i1, i2)] = v_corr_idxs[keep_mask]
        total_after += int(keep_mask.sum())

    # Drop pairs left too weak after re-scoring (GLOMAP FilterInlierNum + FilterInlierRatio).
    filtered_corr, filtered_R, filtered_U = {}, {}, {}
    num_pairs_removed = 0
    for (i1, i2), corr in rescored.items():
        original_count = len(v_corr_idxs_dict.get((i1, i2), []))
        if len(corr) >= min_inlier_count and len(corr) / max(original_count, 1) >= min_inlier_ratio:
            filtered_corr[(i1, i2)] = corr
            if (i1, i2) in i2Ri1_dict:
                filtered_R[(i1, i2)] = i2Ri1_dict[(i1, i2)]
            if (i1, i2) in i2Ui1_dict:
                filtered_U[(i1, i2)] = i2Ui1_dict[(i1, i2)]
        else:
            num_pairs_removed += 1

    logger.info(
        "F-matrix inlier re-scoring: %d → %d correspondences across %d edges (%.1f%% kept). "
        "Removed %d weak pairs (<%d inliers or <%.0f%% ratio), %d pairs remain.",
        total_before, total_after, len(v_corr_idxs_dict),
        100.0 * total_after / max(total_before, 1),
        num_pairs_removed, min_inlier_count, min_inlier_ratio * 100, len(filtered_corr),
    )
    return filtered_corr, filtered_R, filtered_U


def _filter_edges(
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    v_corr_idxs_dict: AnnotatedGraph[np.ndarray],
    edges_to_remove: set,
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3], AnnotatedGraph[np.ndarray]]:
    """Remove edges flagged by view graph calibration."""
    if not edges_to_remove:
        return i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict
    filtered_R = {k: v for k, v in i2Ri1_dict.items() if k not in edges_to_remove}
    filtered_U = {k: v for k, v in i2Ui1_dict.items() if k not in edges_to_remove}
    filtered_corr = {k: v for k, v in v_corr_idxs_dict.items() if k not in edges_to_remove}
    logger.info("Edge filtering: removed %d edges, %d remain.", len(edges_to_remove), len(filtered_R))
    return filtered_R, filtered_U, filtered_corr


def _sync_two_view_reports_after_calibration(
    two_view_reports: AnnotatedGraph[TwoViewEstimationReport],
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    v_corr_idxs_dict: AnnotatedGraph[np.ndarray],
) -> AnnotatedGraph[TwoViewEstimationReport]:
    """Keep view-graph reports consistent with the calibrated edge set and poses.

    After view-graph calibration, edges may be removed and surviving edges may get
    updated relative poses. The reports returned from the earlier view-graph estimator
    should reflect that final state.
    """
    synced_reports: AnnotatedGraph[TwoViewEstimationReport] = {}

    for edge, report in two_view_reports.items():
        if report is None or edge not in i2Ri1_dict or edge not in i2Ui1_dict or edge not in v_corr_idxs_dict:
            continue

        v_corr_idxs = v_corr_idxs_dict[edge]
        synced_reports[edge] = dataclasses.replace(
            report,
            v_corr_idxs=v_corr_idxs,
            num_inliers_est_model=v_corr_idxs.shape[0],
            i2Ri1=i2Ri1_dict[edge],
            i2Ui1=i2Ui1_dict[edge],
        )

    return synced_reports


def reestimate_relative_poses(
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    v_corr_idxs_dict: AnnotatedGraph[np.ndarray],
    keypoints_list: List[Keypoints],
    intrinsics: List[gtsfm_types.CALIBRATION_TYPE],
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3]]:
    """Re-estimate relative poses using refined intrinsics.

    After view graph calibration improves focal lengths, re-compute E-matrices
    and decompose into relative rotation/translation using the updated intrinsics.

    Args:
        i2Ri1_dict: Current relative rotations.
        i2Ui1_dict: Current relative translation directions.
        v_corr_idxs_dict: Verified correspondence indices per image pair.
        keypoints_list: Keypoints for all images.
        intrinsics: Refined intrinsics from view graph calibration.

    Returns:
        Updated relative rotations and translation directions.
    """

    updated_i2Ri1 = {}
    updated_i2Ui1 = {}
    num_updated = 0
    num_failed = 0

    for i1, i2 in i2Ri1_dict:
        if (i1, i2) not in v_corr_idxs_dict:
            updated_i2Ri1[(i1, i2)] = i2Ri1_dict[(i1, i2)]
            updated_i2Ui1[(i1, i2)] = i2Ui1_dict[(i1, i2)]
            continue

        v_corr_idxs = v_corr_idxs_dict[(i1, i2)]
        if v_corr_idxs.shape[0] < 5:
            updated_i2Ri1[(i1, i2)] = i2Ri1_dict[(i1, i2)]
            updated_i2Ui1[(i1, i2)] = i2Ui1_dict[(i1, i2)]
            continue

        coords_i1 = keypoints_list[i1].coordinates[v_corr_idxs[:, 0]]
        coords_i2 = keypoints_list[i2].coordinates[v_corr_idxs[:, 1]]

        K1 = intrinsics[i1]
        K2 = intrinsics[i2]

        # Estimate F-matrix from verified correspondences.
        F, mask = cv2.findFundamentalMat(coords_i1, coords_i2, method=cv2.FM_8POINT)
        if F is None or F.shape != (3, 3):
            updated_i2Ri1[(i1, i2)] = i2Ri1_dict[(i1, i2)]
            updated_i2Ui1[(i1, i2)] = i2Ui1_dict[(i1, i2)]
            num_failed += 1
            continue

        # Convert F → E using refined intrinsics, then decompose.
        i2Ei1 = verification_utils.fundamental_to_essential_matrix(F, K1, K2)
        i2Ri1_new, i2Ui1_new = verification_utils.recover_relative_pose_from_essential_matrix(
            i2Ei1,
            coords_i1,
            coords_i2,
            K1,
            K2,
        )

        if i2Ri1_new is not None and i2Ui1_new is not None:
            updated_i2Ri1[(i1, i2)] = i2Ri1_new
            updated_i2Ui1[(i1, i2)] = i2Ui1_new
            num_updated += 1
        else:
            updated_i2Ri1[(i1, i2)] = i2Ri1_dict[(i1, i2)]
            updated_i2Ui1[(i1, i2)] = i2Ui1_dict[(i1, i2)]
            num_failed += 1

    logger.info(
        "Re-estimated relative poses: %d updated, %d failed, %d total.",
        num_updated,
        num_failed,
        len(i2Ri1_dict),
    )
    return updated_i2Ri1, updated_i2Ui1


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


def get_2d_tracks(correspondences: AnnotatedGraph[np.ndarray], keypoints_graph: List[Keypoints]) -> List[SfmTrack2d]:
    tracks_estimator = CppDsfTracksEstimator()
    return tracks_estimator.run(correspondences, keypoints_graph)


def filter_corr_by_idx(correspondences: AnnotatedGraph[np.ndarray], idxs: List[Tuple[int, int]]):
    """Filter correspondences by indices.

    Args:
        correspondences: Correspondences as a dictionary.
        idxs: Indices to filter by.

    Returns:
        Filtered correspondences.
    """
    return {k: v for k, v in correspondences.items() if k in idxs}
