"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import dask
import gtsam
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dask.delayed import Delayed
from gtsam import Pose3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.viz as viz_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.feature_extractor import FeatureExtractor
from gtsfm.multi_view_optimizer import MultiViewOptimizer
from gtsfm.two_view_estimator import TwoViewEstimator

# paths for storage
PLOT_PATH = "plots"
PLOT_CORRESPONDENCE_PATH = os.path.join(PLOT_PATH, "correspondences")
METRICS_PATH = "result_metrics"
RESULTS_PATH = "results"

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


class SceneOptimizer:
    """Wrapper combining different modules to run the whole pipeline on a
    loader."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        two_view_estimator: TwoViewEstimator,
        multiview_optimizer: MultiViewOptimizer,
        save_viz: bool,
        save_bal_files: bool,
        pose_angular_error_thresh: float,
    ) -> None:
        """ pose_angular_error_thresh is given in degrees """
        self.feature_extractor = feature_extractor
        self.two_view_estimator = two_view_estimator
        self.multiview_optimizer = multiview_optimizer

        self._save_viz = save_viz
        self._save_bal_files = save_bal_files
        self._pose_angular_error_thresh = pose_angular_error_thresh

        # make directories for persisting data
        os.makedirs(PLOT_PATH, exist_ok=True)
        os.makedirs(PLOT_CORRESPONDENCE_PATH, exist_ok=True)
        os.makedirs(METRICS_PATH, exist_ok=True)
        os.makedirs(RESULTS_PATH, exist_ok=True)

    def create_computation_graph(
        self,
        num_images: int,
        image_pair_indices: List[Tuple[int, int]],
        image_graph: List[Delayed],
        camera_intrinsics_graph: List[Delayed],
        use_intrinsics_in_verification: bool = True,
        gt_pose_graph: Optional[List[Delayed]] = None,
    ) -> Delayed:
        """The SceneOptimizer plate calls the FeatureExtractor and TwoViewEstimator plates several times."""

        # auxiliary graph elements for visualizations and saving intermediate
        # data for analysis, not returned to the user.
        auxiliary_graph_list = []

        # detection and description graph
        keypoints_graph_list = []
        descriptors_graph_list = []
        for delayed_image in image_graph:
            (delayed_dets, delayed_descs,) = self.feature_extractor.create_computation_graph(delayed_image)
            keypoints_graph_list += [delayed_dets]
            descriptors_graph_list += [delayed_descs]

        # estimate two-view geometry and get indices of verified correspondences.
        i2Ri1_graph_dict = {}
        i2Ui1_graph_dict = {}
        v_corr_idxs_graph_dict = {}

        frontend_metrics_dict = {}

        for (i1, i2) in image_pair_indices:
            if gt_pose_graph is not None:
                gt_relative_pose = dask.delayed(lambda x, y: x.between(y))(gt_pose_graph[i2], gt_pose_graph[i1])
            else:
                gt_relative_pose = None

            (
                i2Ri1,
                i2Ui1,
                v_corr_idxs,
                rot_error,
                unit_tran_error,
                correspondence_stats,
            ) = self.two_view_estimator.create_computation_graph(
                keypoints_graph_list[i1],
                keypoints_graph_list[i2],
                descriptors_graph_list[i1],
                descriptors_graph_list[i2],
                camera_intrinsics_graph[i1],
                camera_intrinsics_graph[i2],
                use_intrinsics_in_verification,
                gt_relative_pose,
            )
            i2Ri1_graph_dict[(i1, i2)] = i2Ri1
            i2Ui1_graph_dict[(i1, i2)] = i2Ui1
            v_corr_idxs_graph_dict[(i1, i2)] = v_corr_idxs

            if gt_pose_graph is not None:
                frontend_metrics_dict[(i1, i2)] = (
                    rot_error,
                    unit_tran_error,
                    correspondence_stats[0],
                    correspondence_stats[1],
                )

            if self._save_viz:
                auxiliary_graph_list.append(
                    dask.delayed(visualize_twoview_correspondences)(
                        image_graph[i1],
                        image_graph[i2],
                        keypoints_graph_list[i1],
                        keypoints_graph_list[i2],
                        v_corr_idxs,
                        os.path.join(PLOT_CORRESPONDENCE_PATH, "{}_{}.jpg".format(i1, i2)),
                    )
                )

        # persist all front-end metrics and its summary
        if gt_pose_graph is not None:
            auxiliary_graph_list.append(dask.delayed(persist_frontend_metrics_full)(frontend_metrics_dict))

            auxiliary_graph_list.append(
                dask.delayed(aggregate_frontend_metrics)(frontend_metrics_dict, self._pose_angular_error_thresh,)
            )

        # as visualization tasks are not to be provided to the user, we create a
        # dummy computation of concatenating viz tasks with the output graph,
        # forcing computation of viz tasks. Doing this here forces the
        # frontend's auxiliary tasks to be computed before the multi-view stage.
        keypoints_graph_list = dask.delayed(lambda x, y: (x, y))(keypoints_graph_list, auxiliary_graph_list)[0]
        auxiliary_graph_list = []

        (ba_input_graph, ba_output_graph, optimizer_metrics_graph,) = self.multiview_optimizer.create_computation_graph(
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

        filtered_sfm_data_graph = dask.delayed(ba_output_graph.filter_landmarks)(
            self.multiview_optimizer.data_association_module.reproj_error_thresh
        )

        if self._save_viz:
            os.makedirs(os.path.join(PLOT_PATH, "ba_input"), exist_ok=True)
            os.makedirs(os.path.join(PLOT_PATH, "results"), exist_ok=True)

            auxiliary_graph_list.append(
                dask.delayed(visualize_sfm_data)(ba_input_graph, os.path.join(PLOT_PATH, "ba_input"))
            )

            auxiliary_graph_list.append(
                dask.delayed(visualize_sfm_data)(filtered_sfm_data_graph, os.path.join(PLOT_PATH, "results"))
            )

            auxiliary_graph_list.append(
                dask.delayed(visualize_camera_poses)(
                    ba_input_graph, filtered_sfm_data_graph, gt_pose_graph, os.path.join(PLOT_PATH, "results")
                )
            )

        if self._save_bal_files:
            # save the input to Bundle Adjustment (from data association)
            auxiliary_graph_list.append(
                dask.delayed(io_utils.write_cameras)(ba_input_graph, save_dir=os.path.join(RESULTS_PATH, "ba_input"))
            )
            auxiliary_graph_list.append(
                dask.delayed(io_utils.write_images)(ba_input_graph, save_dir=os.path.join(RESULTS_PATH, "ba_input"))
            )
            # save the output of Bundle Adjustment (after optimization)
            auxiliary_graph_list.append(
                dask.delayed(io_utils.write_cameras)(
                    filtered_sfm_data_graph, save_dir=os.path.join(RESULTS_PATH, "ba_output")
                )
            )
            auxiliary_graph_list.append(
                dask.delayed(io_utils.write_images)(
                    filtered_sfm_data_graph, save_dir=os.path.join(RESULTS_PATH, "ba_output")
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
    plot_img = viz_utils.plot_twoview_correspondences(image_i1, image_i2, keypoints_i1, keypoints_i2, corr_idxs_i1i2,)

    io_utils.save_image(plot_img, file_path)


def visualize_sfm_data(sfm_data: GtsfmData, folder_name: str) -> None:
    """Visualize the camera poses and 3d points in SfmData.

    Args:
        sfm_data: data to visualize.
        folder_name: folder to save the visualization at.
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")

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
    pre_ba_sfm_data: GtsfmData, post_ba_sfm_data: GtsfmData, gt_pose_graph: Optional[List[Pose3]], folder_name: str,
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
    ax = fig.gca(projection="3d")

    viz_utils.plot_poses_3d(pre_ba_poses, ax, center_marker_color="c", label_name="Pre-BA")
    viz_utils.plot_poses_3d(post_ba_poses, ax, center_marker_color="k", label_name="Post-BA")
    if gt_pose_graph is not None:
        gt_pose_graph = comp_utils.align_poses_sim3(post_ba_poses, gt_pose_graph)
        viz_utils.plot_poses_3d(gt_pose_graph, ax, center_marker_color="m", label_name="GT")

    ax.legend(loc="upper left")
    viz_utils.set_axes_equal(ax)

    # save the 3D plot in the original view
    fig.savefig(os.path.join(folder_name, "poses_3d.png"))

    # save the BEV representation
    default_camera_elevation = 100  # in metres above ground
    ax.view_init(azim=0, elev=default_camera_elevation)
    fig.savefig(os.path.join(folder_name, "poses_bev.png"))

    plt.close(fig)


def persist_frontend_metrics_full(metrics: Dict[Tuple[int, int], FRONTEND_METRICS_FOR_PAIR],) -> None:
    """Persist the front-end metrics for every pair on disk.

    Args:
        metrics: front-end metrics for pairs of images.
    """

    metrics_list = [
        {
            "i1": k[0],
            "i2": k[1],
            "rotation_angular_error": v[0],
            "translation_angular_error": v[1],
            "num_correct_corr": v[2],
            "inlier_ratio": v[3],
        }
        for k, v in metrics.items()
    ]

    io_utils.save_json_file(os.path.join(METRICS_PATH, "frontend_full.json"), metrics_list)


def aggregate_frontend_metrics(
    metrics: Dict[Tuple[int, int], FRONTEND_METRICS_FOR_PAIR], angular_err_threshold_deg: float
) -> None:
    """Aggregate the front-end metrics to log summary statistics.

    Args:
        metrics: front-end metrics for pairs of images.
        angular_err_threshold_deg: threshold for classifying angular error metrics as success.
    """
    num_entries = len(metrics)

    metrics_array = np.array(list(metrics.values()), dtype=float)

    # count number of rot3 errors which are not None. Should be same in rot3/unit3
    num_valid_entries = np.count_nonzero(~np.isnan(metrics_array[:, 0]))

    # compute pose errors by picking the max error from rot3 and unit3 errors
    pose_errors = np.amax(metrics_array[:, :2], axis=1)

    # check errors against the threshold
    success_count_rot3 = np.sum(metrics_array[:, 0] < angular_err_threshold_deg)
    success_count_unit3 = np.sum(metrics_array[:, 1] < angular_err_threshold_deg)
    success_count_pose = np.sum(pose_errors < angular_err_threshold_deg)

    # count entries with inlier ratio == 1.
    all_correct = np.count_nonzero(metrics_array[:, 3] == 1.0)

    logger.debug(
        "[Two view optimizer] [Summary] Rotation success: %d/%d/%d", success_count_rot3, num_valid_entries, num_entries,
    )

    logger.debug(
        "[Two view optimizer] [Summary] Translation success: %d/%d/%d",
        success_count_unit3,
        num_valid_entries,
        num_entries,
    )

    logger.debug(
        "[Two view optimizer] [Summary] Pose success: %d/%d/%d", success_count_pose, num_valid_entries, num_entries,
    )

    logger.debug(
        "[Two view optimizer] [Summary] Image pairs with 100%% inlier ratio:: %d/%d", all_correct, num_entries,
    )

    front_end_result_info = {
        "angular_err_threshold_deg": angular_err_threshold_deg,
        "num_valid_entries": int(num_valid_entries),
        "num_total_entries": int(num_entries),
        "rotation": {"success_count": int(success_count_rot3),},
        "translation": {"success_count": int(success_count_unit3),},
        "pose": {"success_count": int(success_count_pose),},
        "correspondences": {"all_inliers": int(all_correct),},
    }

    io_utils.save_json_file(os.path.join(METRICS_PATH, "frontend_summary.json"), front_end_result_info)
