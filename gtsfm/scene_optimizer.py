"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""
import logging
import os
from typing import Any, List, Optional, Tuple

import dask
import gtsam
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dask.delayed import Delayed
from gtsam import (
    Pose3,
    SfmData,
    Unit3,
)

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.serialization  # import needed to register serialization fns
import gtsfm.utils.viz as viz_utils
from gtsfm.averaging.rotation.rotation_averaging_base import (
    RotationAveragingBase,
)
from gtsfm.averaging.translation.translation_averaging_base import (
    TranslationAveragingBase,
)
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.feature_extractor import FeatureExtractor
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import (
    DetectorDescriptorBase,
)
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.verifier.verifier_base import VerifierBase
from gtsfm.multi_view_optimizer import MultiViewOptimizer
from gtsfm.two_view_estimator import TwoViewEstimator

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
        pose_angular_error_thresh: float
    ) -> None:
        """ pose_angular_error_thresh is given in degrees """
        self.feature_extractor = feature_extractor
        self.two_view_estimator = two_view_estimator
        self.multiview_optimizer = multiview_optimizer

        self._save_viz = save_viz
        self._save_bal_files = save_bal_files
        self._pose_angular_error_thresh = pose_angular_error_thresh

    def create_computation_graph(
        self,
        num_images: int,
        image_pair_indices: List[Tuple[int, int]],
        image_graph: List[Delayed],
        camera_intrinsics_graph: List[Delayed],
        use_intrinsics_in_verification: bool = True,
        gt_pose_graph: Optional[List[Delayed]] = None,
    ) -> Delayed:
        """ The SceneOptimizer plate calls the FeatureExtractor and TwoViewEstimator plates several times"""

        # auxiliary graph elements for visualizations and saving intermediate
        # data for analysis, not returned to the user.
        auxiliary_graph_list = []

        # detection and description graph
        keypoints_graph_list = []
        descriptors_graph_list = []
        for delayed_image in image_graph:
            (
                delayed_dets,
                delayed_descs,
            ) = self.feature_extractor.create_computation_graph(delayed_image)
            keypoints_graph_list += [delayed_dets]
            descriptors_graph_list += [delayed_descs]

        # estimate two-view geometry and get indices of verified correspondences.
        i2Ri1_graph_dict = {}
        i2Ui1_graph_dict = {}
        v_corr_idxs_graph_dict = {}

        frontend_rot3_errors = []
        frontend_unit3_errors = []

        for (i1, i2) in image_pair_indices:
            if gt_pose_graph is not None:
                gt_relative_pose = dask.delayed(lambda x, y: x.between(y))(
                    gt_pose_graph[i2], gt_pose_graph[i1]
                )
            else:
                gt_relative_pose = None

            (
                i2Ri1,
                i2Ui1,
                v_corr_idxs,
                rot_error,
                unit_tran_error,
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
                frontend_rot3_errors.append(rot_error)
                frontend_unit3_errors.append(unit_tran_error)

            if self._save_viz:
                os.makedirs("plots/correspondences", exist_ok=True)
                auxiliary_graph_list.append(
                    dask.delayed(visualize_twoview_correspondences)(
                        image_graph[i1],
                        image_graph[i2],
                        keypoints_graph_list[i1],
                        keypoints_graph_list[i2],
                        v_corr_idxs,
                        "plots/correspondences/{}_{}.jpg".format(i1, i2),
                    )
                )

        # aggregate metrics for frontend
        if gt_pose_graph is not None:
            auxiliary_graph_list.append(
                dask.delayed(aggregate_frontend_metrics)(
                    frontend_rot3_errors,
                    frontend_unit3_errors,
                    self._pose_angular_error_thresh,
                )
            )
        # as visualization tasks are not to be provided to the user, we create a
        # dummy computation of concatenating viz tasks with the output graph,
        # forcing computation of viz tasks. Doing this here forces the
        # frontend's auxiliary tasks to be computed before the multi-view stage.
        keypoints_graph_list = dask.delayed(lambda x, y: (x, y))(
            keypoints_graph_list, auxiliary_graph_list
        )[0]
        auxiliary_graph_list = []

        (
            ba_input_graph,
            ba_output_graph,
        ) = self.multiview_optimizer.create_computation_graph(
            num_images,
            keypoints_graph_list,
            i2Ri1_graph_dict,
            i2Ui1_graph_dict,
            v_corr_idxs_graph_dict,
            camera_intrinsics_graph,
        )

        filtered_sfm_data_graph = dask.delayed(
            ba_output_graph.filter_landmarks
        )(self.multiview_optimizer.data_association_module.reproj_error_thresh)

        if self._save_viz:
            os.makedirs("plots/ba_input", exist_ok=True)
            os.makedirs("plots/results", exist_ok=True)

            auxiliary_graph_list.append(
                dask.delayed(visualize_sfm_data)(
                    ba_input_graph, "plots/ba_input/"
                )
            )

            auxiliary_graph_list.append(
                dask.delayed(visualize_sfm_data)(
                    filtered_sfm_data_graph, "plots/results/"
                )
            )

            auxiliary_graph_list.append(
                dask.delayed(visualize_camera_poses)(
                    ba_input_graph,
                    filtered_sfm_data_graph,
                    gt_pose_graph,
                    "plots/results",
                )
            )

        if self._save_bal_files:
            os.makedirs("results", exist_ok=True)
            # save the input to Bundle Adjustment (from data association)
            auxiliary_graph_list.append(
                dask.delayed(write_sfmdata_to_disk)(
                    ba_input_graph, "results/ba_input.bal"
                )
            )
            # save the output of Bundle Adjustment (after optimization)
            auxiliary_graph_list.append(
                dask.delayed(write_sfmdata_to_disk)(
                    filtered_sfm_data_graph, "results/ba_output.bal"
                )
            )

        # as visualization tasks are not to be provided to the user, we create a
        # dummy computation of concatenating viz tasks with the output graph,
        # forcing computation of viz tasks
        output_graph = dask.delayed(lambda x, y: (x, y))(
            ba_output_graph, auxiliary_graph_list
        )

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
    plot_img = viz_utils.plot_twoview_correspondences(
        image_i1,
        image_i2,
        keypoints_i1,
        keypoints_i2,
        corr_idxs_i1i2,
    )

    io_utils.save_image(plot_img, file_path)


def visualize_sfm_data(sfm_data: SfmData, folder_name: str) -> None:
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
    pre_ba_sfm_data: SfmData,
    post_ba_sfm_data: SfmData,
    gt_pose_graph: Optional[List[Pose3]],
    folder_name: str,
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
    for i in range(pre_ba_sfm_data.number_cameras()):
        pre_ba_poses.append(pre_ba_sfm_data.camera(i).pose())

    post_ba_poses = []
    for i in range(post_ba_sfm_data.number_cameras()):
        post_ba_poses.append(post_ba_sfm_data.camera(i).pose())

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    viz_utils.plot_poses_3d(pre_ba_poses, ax, center_marker_color="c")
    viz_utils.plot_poses_3d(post_ba_poses, ax, center_marker_color="k")
    if gt_pose_graph is not None:
        gt_pose_graph = comp_utils.align_poses(gt_pose_graph, post_ba_poses)
        viz_utils.plot_poses_3d(gt_pose_graph, ax, center_marker_color="m")

    # save the 3D plot in the original view
    fig.savefig(os.path.join(folder_name, "poses_3d.png"))

    # save the BEV representation
    default_camera_elevation = 100  # in metres above ground
    ax.view_init(azim=0, elev=default_camera_elevation)
    fig.savefig(os.path.join(folder_name, "poses_bev.png"))

    plt.close(fig)


def write_sfmdata_to_disk(sfm_data: SfmData, save_fpath: str) -> None:
    """Write SfmData object as a "Bundle Adjustment in the Large" (BAL) file
    See https://grail.cs.washington.edu/projects/bal/ for more details on the format.

    Note: Need this wrapper as dask cannot directly work on gtsam function
    calls.

    Args:
        sfm_data: data to write.
        save_fpath: filepath to save the data at.
    """
    gtsam.writeBAL(save_fpath, sfm_data)


def aggregate_frontend_metrics(
    rot3_errors: List[Optional[float]],
    unit3_errors: List[Optional[float]],
    angular_err_threshold_deg: float,
) -> None:
    """Aggregate the front-end metrics to log summary statistics.

    Args:
        rot3_errors: angular errors in rotations.
        unit3_errors: angular errors in unit-translations.
        angular_err_threshold: threshold to classify the error as success.
    """
    angular_err_threshold_rad = np.deg2rad(angular_err_threshold_deg)
    num_entries = len(rot3_errors)

    rot3_errors = np.array(rot3_errors, dtype=float)
    unit3_errors = np.array(unit3_errors, dtype=float)

    # count number of entries not nan. Should be same in rot3/unit3
    num_valid_entries = np.count_nonzero(~np.isnan(rot3_errors))

    # compute pose errors by picking the max error
    pose_errors = np.maximum(rot3_errors, unit3_errors)

    # check errors against the threshold
    success_count_rot3 = np.sum(rot3_errors < angular_err_threshold_rad)
    success_count_unit3 = np.sum(unit3_errors < angular_err_threshold_rad)
    success_count_pose = np.sum(pose_errors < angular_err_threshold_rad)

    logger.debug(
        "[Two view optimizer] [Summary] Rotation success: %d/%d/%d",
        success_count_rot3,
        num_valid_entries,
        num_entries,
    )

    logger.debug(
        "[Two view optimizer] [Summary] Translation success: %d/%d/%d",
        success_count_unit3,
        num_valid_entries,
        num_entries,
    )

    logger.debug(
        "[Two view optimizer] [Summary] Pose success: %d/%d/%d",
        success_count_pose,
        num_valid_entries,
        num_entries,
    )

