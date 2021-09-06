"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dask
import matplotlib

matplotlib.use("Agg")

from dask.delayed import Delayed

import gtsfm.averaging.rotation.cycle_consistency as cycle_consistency
import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.two_view_estimator as two_view_estimator
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.viz as viz_utils
from gtsfm.common.image import Image
from gtsfm.feature_extractor import FeatureExtractor
from gtsfm.multi_view_optimizer import MultiViewOptimizer
from gtsfm.two_view_estimator import TwoViewEstimator, TwoViewEstimationReport

# base paths for storage
PLOT_BASE_PATH = Path(__file__).resolve().parent.parent / "plots"
METRICS_PATH = Path(__file__).resolve().parent.parent / "result_metrics"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results"

# plot baths
PLOT_CORRESPONDENCE_PATH = PLOT_BASE_PATH / "correspondences"
PLOT_BA_INPUT_PATH = PLOT_BASE_PATH / "ba_input"
PLOT_RESULTS_PATH = PLOT_BASE_PATH / "results"


# Paths to Save Output in React Folders.
REACT_METRICS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "src" / "result_metrics"
REACT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "public" / "results"

logger = logger_utils.get_logger()

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

# number of digits (significant figures) to include in each entry of error metrics
PRINT_NUM_SIG_FIGS = 2


class SceneOptimizer:
    """Wrapper combining different modules to run the whole pipeline on a
    loader."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        two_view_estimator: TwoViewEstimator,
        multiview_optimizer: MultiViewOptimizer,
        save_two_view_correspondences_viz: bool,
        save_3d_viz: bool,
        save_gtsfm_data: bool,
        pose_angular_error_thresh: float,
    ) -> None:
        """pose_angular_error_thresh is given in degrees"""
        self.feature_extractor = feature_extractor
        self.two_view_estimator = two_view_estimator
        self.multiview_optimizer = multiview_optimizer

        self._save_two_view_correspondences_viz = save_two_view_correspondences_viz
        self._save_3d_viz = save_3d_viz

        self._save_gtsfm_data = save_gtsfm_data
        self._pose_angular_error_thresh = pose_angular_error_thresh

        # make directories for persisting data
        os.makedirs(PLOT_BASE_PATH, exist_ok=True)
        os.makedirs(METRICS_PATH, exist_ok=True)
        os.makedirs(RESULTS_PATH, exist_ok=True)

        os.makedirs(PLOT_CORRESPONDENCE_PATH, exist_ok=True)
        os.makedirs(PLOT_BA_INPUT_PATH, exist_ok=True)
        os.makedirs(PLOT_RESULTS_PATH, exist_ok=True)

        # Save duplicate directories within React folders.
        os.makedirs(REACT_RESULTS_PATH, exist_ok=True)
        os.makedirs(REACT_METRICS_PATH, exist_ok=True)

    def create_computation_graph(
        self,
        num_images: int,
        image_pair_indices: List[Tuple[int, int]],
        image_graph: List[Delayed],
        camera_intrinsics_graph: List[Delayed],
        image_shape_graph: List[Delayed],
        gt_pose_graph: Optional[List[Delayed]] = None,
    ) -> Delayed:
        """The SceneOptimizer plate calls the FeatureExtractor and TwoViewEstimator plates several times."""

        # auxiliary graph elements for visualizations and saving intermediate
        # data for analysis, not returned to the user.
        auxiliary_graph_list = []
        metrics_graph_list = []

        # detection and description graph
        keypoints_graph_list = []
        descriptors_graph_list = []
        for delayed_image in image_graph:
            (delayed_dets, delayed_descs) = self.feature_extractor.create_computation_graph(delayed_image)
            keypoints_graph_list += [delayed_dets]
            descriptors_graph_list += [delayed_descs]

        # estimate two-view geometry and get indices of verified correspondences.
        i2Ri1_graph_dict = {}
        i2Ui1_graph_dict = {}
        v_corr_idxs_graph_dict = {}

        two_view_reports_dict = {}

        for (i1, i2) in image_pair_indices:
            if gt_pose_graph is not None:
                # compute GT relative pose
                gt_i2Ti1 = dask.delayed(lambda x, y: x.between(y))(gt_pose_graph[i2], gt_pose_graph[i1])
            else:
                gt_i2Ti1 = None

            (i2Ri1, i2Ui1, v_corr_idxs, two_view_report,) = self.two_view_estimator.create_computation_graph(
                keypoints_graph_list[i1],
                keypoints_graph_list[i2],
                descriptors_graph_list[i1],
                descriptors_graph_list[i2],
                camera_intrinsics_graph[i1],
                camera_intrinsics_graph[i2],
                image_shape_graph[i1],
                image_shape_graph[i2],
                gt_i2Ti1,
            )

            i2Ri1_graph_dict[(i1, i2)] = i2Ri1
            i2Ui1_graph_dict[(i1, i2)] = i2Ui1
            v_corr_idxs_graph_dict[(i1, i2)] = v_corr_idxs

            two_view_reports_dict[(i1, i2)] = two_view_report

            if self._save_two_view_correspondences_viz:
                auxiliary_graph_list.append(
                    dask.delayed(viz_utils.save_twoview_correspondences_viz)(
                        image_graph[i1],
                        image_graph[i2],
                        keypoints_graph_list[i1],
                        keypoints_graph_list[i2],
                        v_corr_idxs,
                        file_path=os.path.join(PLOT_CORRESPONDENCE_PATH, f"{i1}_{i2}.jpg"),
                    )
                )

        # persist all front-end metrics and its summary
        auxiliary_graph_list.append(dask.delayed(save_full_frontend_metrics)(two_view_reports_dict, image_graph))
        if gt_pose_graph is not None:
            metrics_graph_list.append(
                dask.delayed(two_view_estimator.aggregate_frontend_metrics)(
                    two_view_reports_dict, self._pose_angular_error_thresh
                )
            )

        # as visualization tasks are not to be provided to the user, we create a
        # dummy computation of concatenating viz tasks with the output graph,
        # forcing computation of viz tasks. Doing this here forces the
        # frontend's auxiliary tasks to be computed before the multi-view stage.
        keypoints_graph_list = dask.delayed(lambda x, y: (x, y))(keypoints_graph_list, auxiliary_graph_list)[0]
        auxiliary_graph_list = []

        # ensure cycle consistency in triplets
        cycle_consistent_graph = dask.delayed(cycle_consistency.filter_to_cycle_consistent_edges)(
            i2Ri1_graph_dict, i2Ui1_graph_dict, two_view_reports_dict
        )
        i2Ri1_graph_dict = cycle_consistent_graph[0]
        i2Ui1_graph_dict = cycle_consistent_graph[1]

        # TODO (johnwlambert): dump metrics after cycle consistency
        # Note: the MultiviewOptimizer returns BA input and BA output that are aligned to GT via Sim(3).
        (ba_input_graph, ba_output_graph, optimizer_metrics_graph) = self.multiview_optimizer.create_computation_graph(
            image_graph,
            num_images,
            keypoints_graph_list,
            i2Ri1_graph_dict,
            i2Ui1_graph_dict,
            v_corr_idxs_graph_dict,
            camera_intrinsics_graph,
            gt_pose_graph,
        )

        # aggregate metrics for multiview optimizer
        if optimizer_metrics_graph is not None:
            metrics_graph_list.extend(optimizer_metrics_graph)

        # Save metrics to JSON and generate HTML report.
        auxiliary_graph_list.extend(save_metrics_reports(metrics_graph_list))

        if self._save_3d_viz:
            auxiliary_graph_list.extend(save_visualizations(ba_input_graph, ba_output_graph, gt_pose_graph))

        if self._save_gtsfm_data:
            auxiliary_graph_list.extend(save_gtsfm_data(image_graph, ba_input_graph, ba_output_graph))

        # as visualization tasks are not to be provided to the user, we create a
        # dummy computation of concatenating viz tasks with the output graph,
        # forcing computation of viz tasks
        output_graph = dask.delayed(lambda x, y: (x, y))(ba_output_graph, auxiliary_graph_list)

        # return the entry with just the sfm result
        return output_graph[0]


def save_visualizations(
    ba_input_graph: Delayed, ba_output_graph: Delayed, gt_pose_graph: Optional[List[Delayed]]
) -> List[Delayed]:
    """Save SfmData before and after bundle adjustment and camera poses for visualization.

    Accepts delayed GtsfmData before and after bundle adjustment, along with GT poses,
    saves them and returns a delayed object.

    Args:
        ba_input_graph: Delayed GtsfmData input to bundle adjustment.
        ba_output_graph: Delayed GtsfmData output from bundle adjustment.
        gt_pose_graph: Delayed ground truth poses.

    Returns:
        A list of Delayed objects after saving the different visualizations.
    """
    viz_graph_list = []
    viz_graph_list.append(dask.delayed(viz_utils.save_sfm_data_viz)(ba_input_graph, PLOT_BA_INPUT_PATH))
    viz_graph_list.append(dask.delayed(viz_utils.save_sfm_data_viz)(ba_output_graph, PLOT_RESULTS_PATH))
    viz_graph_list.append(
        dask.delayed(viz_utils.save_camera_poses_viz)(ba_input_graph, ba_output_graph, gt_pose_graph, PLOT_RESULTS_PATH)
    )
    return viz_graph_list


def save_gtsfm_data(image_graph: Delayed, ba_input_graph: Delayed, ba_output_graph: Delayed) -> List[Delayed]:
    """Saves the Gtsfm data before and after bundle adjustment.

    Args:
        image_graph: input image wrapped as Delayed objects.
        ba_input_graph: GtsfmData input to bundle adjustment wrapped as Delayed.
        ba_output_graph: GtsfmData output to bundle adjustment wrapped as Delayed.

    Returns:
        A list of delayed objects after saving the input and outputs to bundle adjustment.
    """
    saving_graph_list = []
    # Save a duplicate in REACT_RESULTS_PATH.
    for output_dir in [RESULTS_PATH, REACT_RESULTS_PATH]:
        # Save the input to Bundle Adjustment (from data association).
        saving_graph_list.append(
            dask.delayed(io_utils.export_model_as_colmap_text)(
                ba_input_graph, image_graph, save_dir=os.path.join(output_dir, "ba_input")
            )
        )
        # Save the output of Bundle Adjustment.
        saving_graph_list.append(
            dask.delayed(io_utils.export_model_as_colmap_text)(
                ba_output_graph, image_graph, save_dir=os.path.join(output_dir, "ba_output")
            )
        )
    return saving_graph_list


def save_metrics_reports(metrics_graph_list: Delayed) -> List[Delayed]:
    """Saves metrics to JSON and HTML report.

    Args:
        metrics_graph: List of GtsfmMetricsGroup from different modules wrapped as Delayed.

    Returns:
        List of delayed objects after saving metrics.
    """
    save_metrics_graph_list = []

    if len(metrics_graph_list) == 0:
        return save_metrics_graph_list

    # Save metrics to JSON
    save_metrics_graph_list.append(dask.delayed(metrics_utils.save_metrics_as_json)(metrics_graph_list, METRICS_PATH))
    save_metrics_graph_list.append(
        dask.delayed(metrics_utils.save_metrics_as_json)(metrics_graph_list, REACT_METRICS_PATH)
    )
    save_metrics_graph_list.append(
        dask.delayed(metrics_report.generate_metrics_report_html)(
            metrics_graph_list, os.path.join(METRICS_PATH, "gtsfm_metrics_report.html")
        )
    )
    return save_metrics_graph_list


def save_full_frontend_metrics(
    two_view_report_dict: Dict[Tuple[int, int], TwoViewEstimationReport], images: List[Image]
) -> None:
    """Converts the TwoViewEstimationReports for all image pairs to a Dict and saves it as JSON.

    Args:
        two_view_report_dict: front-end metrics for pairs of images.
        images: list of all images for this scene, in order of image/frame index.
    """
    metrics_list = []

    for (i1, i2), report in two_view_report_dict.items():

        # Note: if GT is unknown, then R_error_deg, U_error_deg, and inlier_ratio_gt_model will be None
        metrics_list.append(
            {
                "i1": i1,
                "i2": i2,
                "i1_filename": images[i1].file_name,
                "i2_filename": images[i2].file_name,
                "rotation_angular_error": round(report.R_error_deg, PRINT_NUM_SIG_FIGS) if report.R_error_deg else None,
                "translation_angular_error": round(report.U_error_deg, PRINT_NUM_SIG_FIGS)
                if report.U_error_deg
                else None,
                "num_inliers_gt_model": report.num_inliers_gt_model if report.num_inliers_gt_model else None,
                "inlier_ratio_gt_model": round(report.inlier_ratio_gt_model, PRINT_NUM_SIG_FIGS)
                if report.inlier_ratio_gt_model
                else None,
                "inlier_ratio_est_model": round(report.inlier_ratio_est_model, PRINT_NUM_SIG_FIGS),
                "num_inliers_est_model": report.num_inliers_est_model,
            }
        )

    io_utils.save_json_file(os.path.join(METRICS_PATH, "frontend_full.json"), metrics_list)

    # Save duplicate copy of 'frontend_full.json' within React Folder.
    io_utils.save_json_file(os.path.join(REACT_METRICS_PATH, "frontend_full.json"), metrics_list)
