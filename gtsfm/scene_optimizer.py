"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""
import copy
import logging
import os
import timeit
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dask
import matplotlib
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dask.delayed import Delayed
from gtsam import Pose3, Rot3, Unit3, Cal3Bundler

import gtsfm.averaging.rotation.cycle_consistency as cycle_utils
import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.metrics as metric_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.viz as viz_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.feature_extractor import FeatureExtractor
from gtsfm.multi_view_optimizer import MultiViewOptimizer
from gtsfm.two_view_estimator import TwoViewEstimator, TwoViewEstimationReport

# paths for storage
PLOT_PATH = "plots"
PLOT_CORRESPONDENCE_PATH = os.path.join(PLOT_PATH, "correspondences")
METRICS_PATH = Path(__file__).resolve().parent.parent / "result_metrics"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results"

# Paths to Save Output in React Folders.
REACT_METRICS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "src" / "result_metrics"
REACT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "public" / "results"

"""
data type for frontend metrics on a pair of images, containing:
1. rotation angular error
2. translation angular error
3. number of correct correspondences
4. inlier ratio
"""
FRONTEND_METRICS_FOR_PAIR = Tuple[Optional[float], Optional[float], int, float]

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
        gt_scene_mesh: Optional[trimesh.Trimesh] = None
    ) -> Delayed:
        """The SceneOptimizer plate calls the FeatureExtractor and TwoViewEstimator plates several times."""

        # auxiliary graph elements for visualizations and saving intermediate
        # data for analysis, not returned to the user.
        auxiliary_graph_list = []

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
                gt_pose_graph[i1],
                gt_pose_graph[i2],
                gt_scene_mesh,
            )

            # optionally: GENERATE SYNTHETIC matches, using GT pose
            # from gtsam import Unit3
            # i2Ri1 = dask.delayed(lambda T: T.rotation())(gt_i2Ti1)
            # i2Ui1 = dask.delayed(lambda T: Unit3(T.translation()))(gt_i2Ti1)


            i2Ri1_graph_dict[(i1, i2)] = i2Ri1
            i2Ui1_graph_dict[(i1, i2)] = i2Ui1
            v_corr_idxs_graph_dict[(i1, i2)] = v_corr_idxs

            two_view_reports_dict[(i1, i2)] = two_view_report

            # Get inlier mask using ground truth scene mesh
            # if gt_scene_mesh is not None:
            #     inlier_mask = dask.delayed(mesh_inlier_correspondences)(
            #             keypoints_graph_list[i1],
            #             keypoints_graph_list[i2],
            #             v_corr_idxs,
            #             camera_intrinsics_graph[i1],
            #             camera_intrinsics_graph[i2],
            #             gt_pose_graph[i1],
            #             gt_pose_graph[i2],
            #             gt_scene_mesh,
            #     )

            if self._save_two_view_correspondences_viz:                    
                auxiliary_graph_list.append(
                    dask.delayed(visualize_twoview_correspondences)(
                        image_graph[i1],
                        image_graph[i2],
                        keypoints_graph_list[i1],
                        keypoints_graph_list[i2],
                        v_corr_idxs,
                        os.path.join(PLOT_CORRESPONDENCE_PATH, f"{i1}_{i2}.jpg"),
                        inlier_mask=two_view_report.inlier_mask_gt_model
                    )
                )

        # persist all front-end metrics and its summary
        auxiliary_graph_list.append(dask.delayed(persist_frontend_metrics_full)(two_view_reports_dict, image_graph))

        auxiliary_graph_list.append(
            dask.delayed(aggregate_frontend_metrics)(two_view_reports_dict, self._pose_angular_error_thresh)
        )

        # as visualization tasks are not to be provided to the user, we create a
        # dummy computation of concatenating viz tasks with the output graph,
        # forcing computation of viz tasks. Doing this here forces the
        # frontend's auxiliary tasks to be computed before the multi-view stage.
        keypoints_graph_list = dask.delayed(lambda x, y: (x, y))(keypoints_graph_list, auxiliary_graph_list)[0]
        auxiliary_graph_list = []

        # ensure cycle consistency in triplets
        cycle_consistent_graph = dask.delayed(cycle_utils.filter_to_cycle_consistent_edges)(i2Ri1_graph_dict, i2Ui1_graph_dict, two_view_reports_dict)

        i2Ri1_graph_dict = cycle_consistent_graph[0]
        i2Ui1_graph_dict = cycle_consistent_graph[1]

        (
            ba_input_graph,
            ba_output_graph,
            optimizer_metrics_graph,
            react_metrics_graph,
        ) = self.multiview_optimizer.create_computation_graph(
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
            auxiliary_graph_list.append(optimizer_metrics_graph)

        # add duplicate of optimizer_metrics_graph to save within React file directory
        if react_metrics_graph is not None:
            auxiliary_graph_list.append(react_metrics_graph)

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
    inlier_mask: Optional[np.ndarray] = None,
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
    plot_img = viz_utils.plot_twoview_correspondences(
        image_i1, 
        image_i2, 
        keypoints_i1, 
        keypoints_i2, 
        corr_idxs_i1i2, 
        inlier_mask=inlier_mask
    )

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

        post_ba_pose_errors_dict = metric_utils.compute_pose_errors(
            gt_wTi_list=corresponding_gt_poses, wTi_list=post_ba_poses
        )
        print("post_ba_pose_errors_dict: ", post_ba_pose_errors_dict)

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


def persist_frontend_metrics_full(two_view_report_dict: Dict[Tuple[int, int], TwoViewEstimationReport], images: List[Image]) -> None:
    """Persist the front-end metrics for every pair on disk.

    Args:
        two_view_report_dict: front-end metrics for pairs of images.
    """
    metrics_list = []

    for (i1, i2), report in two_view_report_dict.items():

        if report.i2Ri1:
            qw, qx, qy, qz = report.i2Ri1.quaternion()
            i2ti1 = report.i2Ui1.point3().tolist()

            i2Ri1_coefficients = {"qw": qw, "qx": qx, "qy": qy, "qz": qz}
        else:
            i2Ri1_coefficients = None
            i2ti1 = None
            euler_xyz = None

        # Note: if GT is unknown, then R_error_deg and U_error_deg will be None
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
                "num_H_inliers": int(report.num_H_inliers),
                "H_inlier_ratio": round(report.H_inlier_ratio, PRINT_NUM_SIG_FIGS),
                
                "i2Ri1": i2Ri1_coefficients,
                "i2Ui1": i2ti1
            }
        )  
        

    io_utils.save_json_file(os.path.join(METRICS_PATH, "frontend_full.json"), metrics_list)

    # Save duplicate copy of 'frontend_full.json' within React Folder.
    io_utils.save_json_file(os.path.join(REACT_METRICS_PATH, "frontend_full.json"), metrics_list)


def aggregate_frontend_metrics(
    two_view_report_dict: Dict[Tuple[int, int], TwoViewEstimationReport], angular_err_threshold_deg: float
) -> None:
    """Aggregate the front-end metrics to log summary statistics.

    We define "pose error" as the maximum of the angular errors in rotation and translation.
    References:
        SuperGlue, CVPR 2020: https://arxiv.org/pdf/1911.11763.pdf
        Learning to find good correspondences. CVPR 2018:
        OA-Net, ICCV 2019:
        NG-RANSAC, ICCV 2019:

    Args:
        two_view_report_dict: report containing front-end metrics for each image pair.
        angular_err_threshold_deg: threshold for classifying angular error metrics as success.
    """
    num_entries = len(two_view_report_dict.keys())

    # all rotational errors in degrees
    rot3_angular_errors = []
    trans_angular_errors = []
    avg_inlier_reproj_errors = []
    avg_precisions = []

    for report in two_view_report_dict.values():
        rot3_angular_errors.append(report.R_error_deg)
        trans_angular_errors.append(report.U_error_deg)
        if report.avg_inlier_reproj_err is not None:
            avg_inlier_reproj_errors.append(report.avg_inlier_reproj_err)
        if report.inlier_ratio_gt_model is not None:
            avg_precisions.append(report.inlier_ratio_gt_model)


    rot3_angular_errors = np.array(rot3_angular_errors, dtype=float)
    trans_angular_errors = np.array(trans_angular_errors, dtype=float)

    # count number of rot3 errors which are not None. Should be same in rot3/unit3
    num_valid_entries = np.count_nonzero(~np.isnan(rot3_angular_errors))

    # compute pose errors by picking the max error from rot3 and unit3 errors
    pose_errors = np.maximum(trans_angular_errors, trans_angular_errors)

    # check errors against the threshold
    success_count_rot3 = np.sum(rot3_angular_errors < angular_err_threshold_deg)
    success_count_unit3 = np.sum(trans_angular_errors < angular_err_threshold_deg)
    success_count_pose = np.sum(pose_errors < angular_err_threshold_deg)

    # count entries with inlier ratio == 1.
    # all_correct = np.count_nonzero(metrics_array[:, 3] == 1.0)

    # Note: average of each two view average
    logger.debug(f'[Front end] [Summary] Average precision: {np.mean(avg_precisions)}')
    logger.debug(f'[Front end] [Summary] Average inlier reprojection error: {np.mean(avg_inlier_reproj_errors)}')

    logger.debug(
        "[Two view optimizer] [Summary] Rotation success: %d/%d/%d", success_count_rot3, num_valid_entries, num_entries
    )

    logger.debug(
        "[Two view optimizer] [Summary] Translation success: %d/%d/%d",
        success_count_unit3,
        num_valid_entries,
        num_entries,
    )

    logger.debug(
        "[Two view optimizer] [Summary] Pose success: %d/%d/%d", success_count_pose, num_valid_entries, num_entries
    )

    # logger.debug("[Two view optimizer] [Summary] Image pairs with 100%% inlier ratio:: %d/%d", all_correct, num_entries)

    front_end_result_info = {
        "angular_err_threshold_deg": angular_err_threshold_deg,
        "num_valid_entries": int(num_valid_entries),
        "num_total_entries": int(num_entries),
        "rotation": {"success_count": int(success_count_rot3)},
        "translation": {"success_count": int(success_count_unit3)},
        "pose": {"success_count": int(success_count_pose)},
        # "correspondences": {"all_inliers": int(all_correct)},
    }

    io_utils.save_json_file(os.path.join(METRICS_PATH, "frontend_summary.json"), front_end_result_info)

    # Save duplicate copy of 'frontend_summary.json' within React Folder.
    io_utils.save_json_file(os.path.join(REACT_METRICS_PATH, "frontend_summary.json"), front_end_result_info)
