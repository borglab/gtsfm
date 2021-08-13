"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""
import copy
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dask
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dask.delayed import Delayed
from gtsam import Pose3

import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.viz as viz_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.feature_extractor import FeatureExtractor
from gtsfm.multi_view_optimizer import MultiViewOptimizer
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.two_view_estimator import TwoViewEstimator

# paths for storage
PLOT_PATH = "plots"
PLOT_CORRESPONDENCE_PATH = os.path.join(PLOT_PATH, "correspondences")
METRICS_PATH = Path(__file__).resolve().parent.parent / "result_metrics"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results"

# Paths to Save Output in React Folders.
REACT_METRICS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "src" / "result_metrics"
REACT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "public" / "results"

logger = logger_utils.get_logger()

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)


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
        os.makedirs(PLOT_PATH, exist_ok=True)
        os.makedirs(PLOT_CORRESPONDENCE_PATH, exist_ok=True)
        os.makedirs(METRICS_PATH, exist_ok=True)
        os.makedirs(RESULTS_PATH, exist_ok=True)

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
                    dask.delayed(visualize_twoview_correspondences)(
                        image_graph[i1],
                        image_graph[i2],
                        keypoints_graph_list[i1],
                        keypoints_graph_list[i2],
                        v_corr_idxs,
                        os.path.join(PLOT_CORRESPONDENCE_PATH, f"{i1}_{i2}.jpg"),
                    )
                )

        # persist all front-end metrics and its summary
        auxiliary_graph_list.append(
            dask.delayed(metrics_utils.persist_frontend_metrics_full)(two_view_reports_dict, image_graph)
        )
        if gt_pose_graph is not None:
            metrics_graph_list.append(
                dask.delayed(metrics_utils.aggregate_frontend_metrics)(
                    two_view_reports_dict, self._pose_angular_error_thresh
                )
            )

        # as visualization tasks are not to be provided to the user, we create a
        # dummy computation of concatenating viz tasks with the output graph,
        # forcing computation of viz tasks. Doing this here forces the
        # frontend's auxiliary tasks to be computed before the multi-view stage.
        keypoints_graph_list = dask.delayed(lambda x, y: (x, y))(keypoints_graph_list, auxiliary_graph_list)[0]
        auxiliary_graph_list = []

        (ba_input_graph, ba_output_graph, optimizer_metrics_graph,) = self.multiview_optimizer.create_computation_graph(
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

        # Save metrics to JSON
        auxiliary_graph_list.append(
            dask.delayed(metrics_utils.save_metrics_as_json)(metrics_graph_list, "result_metrics")
        )
        auxiliary_graph_list.append(
            dask.delayed(metrics_utils.save_metrics_as_json)(metrics_graph_list, REACT_METRICS_PATH)
        )
        auxiliary_graph_list.append(
            dask.delayed(metrics_report.generate_metrics_report_html)(
                metrics_graph_list, os.path.join("result_metrics", "gtsfm_metrics_report.html")
            )
        )

        if self._save_3d_viz:
            os.makedirs(os.path.join(PLOT_PATH, "ba_input"), exist_ok=True)
            os.makedirs(os.path.join(PLOT_PATH, "results"), exist_ok=True)

            auxiliary_graph_list.append(
                dask.delayed(visualize_sfm_data)(ba_input_graph, os.path.join(PLOT_PATH, "ba_input"))
            )

            auxiliary_graph_list.append(
                dask.delayed(visualize_sfm_data)(ba_output_graph, os.path.join(PLOT_PATH, "results"))
            )

            auxiliary_graph_list.append(
                dask.delayed(visualize_camera_poses)(
                    ba_input_graph, ba_output_graph, gt_pose_graph, os.path.join(PLOT_PATH, "results")
                )
            )

        if self._save_gtsfm_data:
            # save the input to Bundle Adjustment (from data association)
            ba_input_save_dir = os.path.join(RESULTS_PATH, "ba_input")
            react_ba_input_save_dir = os.path.join(REACT_RESULTS_PATH, "ba_input")
            auxiliary_graph_list.append(
                dask.delayed(io_utils.export_model_as_colmap_text)(
                    ba_input_graph, image_graph, save_dir=ba_input_save_dir
                )
            )

            # Save duplicate copies of input to Bundle Adjustment to React Folder
            auxiliary_graph_list.append(
                dask.delayed(io_utils.export_model_as_colmap_text)(
                    ba_input_graph, image_graph, save_dir=react_ba_input_save_dir
                )
            )

            # save the output of Bundle Adjustment (after optimization)
            ba_output_save_dir = os.path.join(RESULTS_PATH, "ba_output")
            react_ba_output_save_dir = os.path.join(REACT_RESULTS_PATH, "ba_output")
            auxiliary_graph_list.append(
                dask.delayed(io_utils.export_model_as_colmap_text)(
                    ba_output_graph, image_graph, save_dir=ba_output_save_dir
                )
            )

            # Save duplicate copies of output to Bundle Adjustment to React Folder
            auxiliary_graph_list.append(
                dask.delayed(io_utils.export_model_as_colmap_text)(
                    ba_output_graph, image_graph, save_dir=react_ba_output_save_dir
                )
            )

        # as visualization tasks are not to be provided to the user, we create a
        # dummy computation of concatenating viz tasks with the output graph,
        # forcing computation of viz tasks
        output_graph = dask.delayed(lambda x, y: (x, y))(ba_output_graph, auxiliary_graph_list)

        # return the entry with just the sfm result
        return output_graph[0]


def visualize_twoview_correspondences(
    image_i1: Image,
    image_i2: Image,
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    corr_idxs_i1i2: np.ndarray,
    file_path: str,
) -> None:
    """Visualize correspondences between pairs of images.

    Args:
        image_i1: image #i1.
        image_i2: image #i2.
        keypoints_i1: detected Keypoints for image #i1.
        keypoints_i2: detected Keypoints for image #i2.
        corr_idxs_i1i2: correspondence indices.
        file_path: file path to save the visualization.
    """
    plot_img = viz_utils.plot_twoview_correspondences(image_i1, image_i2, keypoints_i1, keypoints_i2, corr_idxs_i1i2)

    io_utils.save_image(plot_img, file_path)


def visualize_sfm_data(sfm_data: GtsfmData, folder_name: str) -> None:
    """Visualize the camera poses and 3d points in SfmData.

    Args:
        sfm_data: data to visualize.
        folder_name: folder to save the visualization at.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    viz_utils.plot_sfm_data_3d(sfm_data, ax)
    viz_utils.set_axes_equal(ax)

    # save the 3D plot in the original view
    fig.savefig(os.path.join(folder_name, "3d.png"))

    # save the BEV representation
    default_camera_elevation = 100  # in metres above ground
    ax.view_init(azim=0, elev=default_camera_elevation)
    fig.savefig(os.path.join(folder_name, "bev.png"))

    plt.close(fig)


def visualize_camera_poses(
    pre_ba_sfm_data: GtsfmData, post_ba_sfm_data: GtsfmData, gt_pose_graph: Optional[List[Pose3]], folder_name: str
) -> None:
    """Visualize the camera pose and save to disk.

    Args:
        pre_ba_sfm_data: data input to bundle adjustment.
        post_ba_sfm_data: output of bundle adjustment.
        gt_pose_graph: ground truth poses.
        folder_name: folder to save the visualization at.
    """
    # extract camera poses
    pre_ba_poses = []
    for i in pre_ba_sfm_data.get_valid_camera_indices():
        pre_ba_poses.append(pre_ba_sfm_data.get_camera(i).pose())

    post_ba_poses = []
    for i in post_ba_sfm_data.get_valid_camera_indices():
        post_ba_poses.append(post_ba_sfm_data.get_camera(i).pose())

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    if gt_pose_graph is not None:
        # Select ground truth poses that correspond to pre-BA and post-BA estimated poses
        # some may have been lost after pruning to largest connected component
        corresponding_gt_poses = [gt_pose_graph[i] for i in pre_ba_sfm_data.get_valid_camera_indices()]

        # ground truth is used as the reference
        pre_ba_poses = comp_utils.align_poses_sim3(corresponding_gt_poses, copy.deepcopy(pre_ba_poses))
        post_ba_poses = comp_utils.align_poses_sim3(corresponding_gt_poses, copy.deepcopy(post_ba_poses))
        viz_utils.plot_poses_3d(gt_pose_graph, ax, center_marker_color="m", label_name="GT")

    viz_utils.plot_poses_3d(pre_ba_poses, ax, center_marker_color="c", label_name="Pre-BA")
    viz_utils.plot_poses_3d(post_ba_poses, ax, center_marker_color="k", label_name="Post-BA")

    ax.legend(loc="upper left")
    viz_utils.set_axes_equal(ax)

    # save the 3D plot in the original view
    fig.savefig(os.path.join(folder_name, "poses_3d.png"))

    # save the BEV representation
    default_camera_elevation = 100  # in metres above ground
    ax.view_init(azim=0, elev=default_camera_elevation)
    fig.savefig(os.path.join(folder_name, "poses_bev.png"))

    plt.close(fig)
