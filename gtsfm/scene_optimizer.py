"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dask
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dask.delayed import Delayed
from gtsam import Pose3, Similarity3, Rot3, Unit3
from trimesh import Trimesh

import gtsfm.common.types as gtsfm_types
import gtsfm.two_view_estimator as two_view_estimator
import gtsfm.utils.ellipsoid as ellipsoid_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.viz as viz_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.densify.mvs_base import MVSBase
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.multi_view_optimizer import MultiViewOptimizer
from gtsfm.retriever.retriever_base import RetrieverBase, ImageMatchingRegime
from gtsfm.two_view_estimator import (
    POST_ISP_REPORT_TAG,
    VIEWGRAPH_REPORT_TAG,
    TwoViewEstimationReport,
    TwoViewEstimator,
)

matplotlib.use("Agg")

DEFAULT_OUTPUT_ROOT = str(Path(__file__).resolve().parent.parent)

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
        retriever: RetrieverBase,
        correspondence_generator: CorrespondenceGeneratorBase,
        two_view_estimator: TwoViewEstimator,
        multiview_optimizer: MultiViewOptimizer,
        dense_multiview_optimizer: Optional[MVSBase] = None,
        save_two_view_correspondences_viz: bool = False,
        save_3d_viz: bool = False,
        save_gtsfm_data: bool = True,
        pose_angular_error_thresh: float = 3,
        output_root: str = DEFAULT_OUTPUT_ROOT,
        output_worker: Optional[str] = None,
    ) -> None:
        """pose_angular_error_thresh is given in degrees"""
        self.retriever = retriever
        self.correspondence_generator = correspondence_generator
        self.two_view_estimator = two_view_estimator
        self.multiview_optimizer = multiview_optimizer
        self.dense_multiview_optimizer = dense_multiview_optimizer

        self._save_two_view_correspondences_viz = save_two_view_correspondences_viz
        self._save_3d_viz = save_3d_viz
        self.run_dense_optimizer = self.dense_multiview_optimizer is not None

        self._save_gtsfm_data = save_gtsfm_data
        self._pose_angular_error_thresh = pose_angular_error_thresh
        self.output_root = Path(output_root)
        self._output_worker = output_worker
        self._create_output_directories()

    def __repr__(self) -> str:
        """Returns string representation of class."""
        return f"""
        {self.retriever}
        {self.correspondence_generator}
        {self.two_view_estimator}
        {self.multiview_optimizer}
        DenseMultiviewOptimizer: {self.dense_multiview_optimizer}
        """

    def _create_output_directories(self) -> None:
        """Create various output directories for GTSFM results, metrics, and plots."""
        # base paths for storage
        self._plot_base_path = self.output_root / "plots"
        self._metrics_path = self.output_root / "result_metrics"
        self._results_path = self.output_root / "results"

        # plot paths
        self._plot_correspondence_path = self._plot_base_path / "correspondences"
        self._plot_ba_input_path = self._plot_base_path / "ba_input"
        self._plot_results_path = self._plot_base_path / "results"
        self._mvs_ply_save_fpath = self._results_path / "mvs_output" / "dense_pointcloud.ply"

        # make directories for persisting data
        os.makedirs(self._plot_base_path, exist_ok=True)
        os.makedirs(self._metrics_path, exist_ok=True)
        os.makedirs(self._results_path, exist_ok=True)

        os.makedirs(self._plot_correspondence_path, exist_ok=True)
        os.makedirs(self._plot_ba_input_path, exist_ok=True)
        os.makedirs(self._plot_results_path, exist_ok=True)

        # Save duplicate directories within React folders.
        os.makedirs(REACT_RESULTS_PATH, exist_ok=True)
        os.makedirs(REACT_METRICS_PATH, exist_ok=True)

    def create_computation_graph(
        self,
        keypoints_list: List[Keypoints],
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport],
        num_images: int,
        images: List[Delayed],
        camera_intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        gt_wTi_list: List[Optional[Pose3]],
        gt_scene_mesh: Optional[Trimesh] = None,
    ) -> Tuple[Delayed, List[Delayed], List[Delayed]]:
        """The SceneOptimizer plate calls the FeatureExtractor and TwoViewEstimator plates several times."""
        logger.info(f"Results, plots, and metrics will be saved at {self.output_root}")

        delayed_results: List[Delayed] = []

        # Note: the MultiviewOptimizer returns BA input and BA output that are aligned to GT via Sim(3).
        (
            ba_input_graph,
            ba_output_graph,
            view_graph_two_view_reports,
            optimizer_metrics_graph,
        ) = self.multiview_optimizer.create_computation_graph(
            images=images,
            num_images=num_images,
            keypoints_list=keypoints_list,
            i2Ri1_dict=i2Ri1_dict,
            i2Ui1_dict=i2Ui1_dict,
            v_corr_idxs_dict=v_corr_idxs_dict,
            all_intrinsics=camera_intrinsics,
            absolute_pose_priors=absolute_pose_priors,
            relative_pose_priors=relative_pose_priors,
            two_view_reports_dict=two_view_reports,
            cameras_gt=cameras_gt,
            gt_wTi_list=gt_wTi_list,
            output_root=self.output_root,
        )
        if view_graph_two_view_reports is not None:
            two_view_reports_post_viewgraph_estimator = view_graph_two_view_reports

        # Persist all front-end metrics and their summaries.
        # TODO(akshay-krishnan): this delays saving the frontend reports until MVO has completed, not ideal.
        metrics_graph_list: List[Delayed] = []
        save_retrieval_metrics = self.retriever._matching_regime in [
            ImageMatchingRegime.RETRIEVAL,
            ImageMatchingRegime.SEQUENTIAL_WITH_RETRIEVAL,
        ]
        annotation = dask.annotate(workers=self._output_worker) if self._output_worker else dask.annotate()
        with annotation:
            delayed_results.append(
                dask.delayed(save_full_frontend_metrics)(
                    two_view_reports,
                    images,
                    filename="two_view_report_{}.json".format(POST_ISP_REPORT_TAG),
                    save_retrieval_metrics=save_retrieval_metrics,
                    metrics_path=self._metrics_path,
                    plot_base_path=self._plot_base_path,
                )
            )

            # TODO(Ayush): pass only image name instead of the whole image. And delete images from memory.
            delayed_results.append(
                dask.delayed(save_full_frontend_metrics)(
                    two_view_reports_post_viewgraph_estimator,
                    images,
                    filename="two_view_report_{}.json".format(VIEWGRAPH_REPORT_TAG),
                    save_retrieval_metrics=save_retrieval_metrics,
                    metrics_path=self._metrics_path,
                    plot_base_path=self._plot_base_path,
                )
            )
            metrics_graph_list.append(
                dask.delayed(two_view_estimator.aggregate_frontend_metrics)(
                    two_view_reports_post_viewgraph_estimator,
                    self._pose_angular_error_thresh,
                    metric_group_name="verifier_summary_{}".format(VIEWGRAPH_REPORT_TAG),
                )
            )

        # aggregate metrics for multiview optimizer
        if optimizer_metrics_graph is not None:
            metrics_graph_list.extend(optimizer_metrics_graph)

        # Modify BA input, BA output, and GT poses to have point clouds and frustums aligned with x,y,z axes.
        ba_input_graph, ba_output_graph, gt_wTi_list = dask.delayed(align_estimated_gtsfm_data, nout=3)(
            ba_input_graph, ba_output_graph, gt_wTi_list
        )

        annotation = dask.annotate(workers=self._output_worker) if self._output_worker else dask.annotate()
        with annotation:
            if self._save_gtsfm_data:
                delayed_results.extend(
                    save_gtsfm_data(
                        images,
                        ba_input_graph,
                        ba_output_graph,
                        results_path=self._results_path,
                        cameras_gt=cameras_gt,
                    )
                )
                if self._save_3d_viz:
                    delayed_results.extend(
                        save_visualizations(
                            aligned_ba_input_graph=ba_input_graph,
                            aligned_ba_output_graph=ba_output_graph,
                            gt_pose_graph=gt_wTi_list,
                            plot_ba_input_path=self._plot_ba_input_path,
                            plot_results_path=self._plot_results_path,
                        )
                    )

        if self.run_dense_optimizer and self.dense_multiview_optimizer is not None:
            img_dict_graph = dask.delayed(get_image_dictionary)(images)
            (
                dense_points_graph,
                dense_point_colors_graph,
                densify_metrics_graph,
                downsampling_metrics_graph,
            ) = self.dense_multiview_optimizer.create_computation_graph(img_dict_graph, ba_output_graph)

            # Cast to string as Open3d cannot use PosixPath's for I/O -- only string file paths are accepted.
            annotation = dask.annotate(workers=self._output_worker) if self._output_worker else dask.annotate()
            with annotation:
                delayed_results.append(
                    dask.delayed(io_utils.save_point_cloud_as_ply)(
                        save_fpath=str(self._mvs_ply_save_fpath),
                        points=dense_points_graph,
                        rgb=dense_point_colors_graph,
                    )
                )

            # Add metrics for dense reconstruction and voxel downsampling
            if densify_metrics_graph is not None:
                metrics_graph_list.append(densify_metrics_graph)
            if downsampling_metrics_graph is not None:
                metrics_graph_list.append(downsampling_metrics_graph)

        # Save metrics to JSON and generate HTML report.
        annotation = dask.annotate(workers=self._output_worker) if self._output_worker else dask.annotate()

        # return the entry with just the sfm result
        return ba_output_graph, delayed_results, metrics_graph_list


def get_image_dictionary(image_list: List[Image]) -> Dict[int, Image]:
    """Convert a list of images to the MVS input format."""
    img_dict = {i: img for i, img in enumerate(image_list)}
    return img_dict


def align_estimated_gtsfm_data(
    ba_input: GtsfmData, ba_output: GtsfmData, gt_wTi_list: List[Optional[Pose3]]
) -> Tuple[GtsfmData, GtsfmData, List[Optional[Pose3]]]:
    """First aligns ba_input and ba_output to gt_wTi_list using a Sim3 transformation, then aligns them all to the
    X, Y, Z axes via another Sim3 global transformation.

    Args:
        ba_input: GtsfmData input to bundle adjustment.
        ba_output: GtsfmData output from bundle adjustment.
        gt_pose_graph: List of GT camera poses.

    Returns:
        Updated ba_input GtsfmData object aligned to axes.
        Updated ba_output GtsfmData object aligned to axes.
        Updated gt_pose_graph with GT poses aligned to axes.
    """
    ba_input = ba_input.align_via_Sim3_to_poses(gt_wTi_list)
    ba_output = ba_output.align_via_Sim3_to_poses(gt_wTi_list)

    walignedTw = ellipsoid_utils.get_ortho_axis_alignment_transform(ba_output)
    walignedSw = Similarity3(R=walignedTw.rotation(), t=walignedTw.translation(), s=1.0)
    ba_input = ba_input.apply_Sim3(walignedSw)
    ba_output = ba_output.apply_Sim3(walignedSw)
    gt_wTi_list = [walignedSw.transformFrom(wTi) if wTi is not None else None for wTi in gt_wTi_list]
    return ba_input, ba_output, gt_wTi_list


def save_visualizations(
    aligned_ba_input_graph: Delayed,
    aligned_ba_output_graph: Delayed,
    gt_pose_graph: List[Optional[Delayed]],
    plot_ba_input_path: Path,
    plot_results_path: Path,
) -> List[Delayed]:
    """Save SfmData before and after bundle adjustment and camera poses for visualization.

    Accepts delayed GtsfmData before and after bundle adjustment, along with GT poses,
    saves them and returns a delayed object.

    Args:
        ba_input_graph: Delayed GtsfmData input to bundle adjustment.
        ba_output_graph: Delayed GtsfmData output from bundle adjustment.
        gt_pose_graph: Delayed ground truth poses.
        plot_ba_input_path: Path to directory where visualizations of bundle adjustment input data will be saved.
        plot_results_path: Path to directory where visualizations of bundle adjustment output data will be saved.

    Returns:
        A list of Delayed objects after saving the different visualizations.
    """
    viz_graph_list = []
    viz_graph_list.append(dask.delayed(viz_utils.save_sfm_data_viz)(aligned_ba_input_graph, plot_ba_input_path))
    viz_graph_list.append(dask.delayed(viz_utils.save_sfm_data_viz)(aligned_ba_output_graph, plot_results_path))
    viz_graph_list.append(
        dask.delayed(viz_utils.save_camera_poses_viz)(
            aligned_ba_input_graph, aligned_ba_output_graph, gt_pose_graph, plot_results_path
        )
    )
    return viz_graph_list


def get_gtsfm_data_with_gt_cameras_and_est_tracks(
    cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
    ba_output: GtsfmData,
) -> GtsfmData:
    """Creates GtsfmData object with GT camera poses and estimated tracks.

    Args:
        gtsfm_data: GtsfmData object with estimated camera poses and tracks.
        cameras_gt: List of GT cameras.

    Returns:
        GtsfmData object with GT camera poses and estimated tracks.
    """
    gt_gtsfm_data = GtsfmData(number_images=len(cameras_gt))
    for i, camera in enumerate(cameras_gt):
        if camera is not None:
            gt_gtsfm_data.add_camera(i, camera)
    for track in ba_output.get_tracks():
        gt_gtsfm_data.add_track(track)
    return gt_gtsfm_data


def save_gtsfm_data(
    image_graph: List[Delayed],
    ba_input_graph: Delayed,
    ba_output_graph: Delayed,
    results_path: Path,
    cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
) -> List[Delayed]:
    """Saves the Gtsfm data before and after bundle adjustment.

    Args:
        image_graph: Input image wrapped as Delayed objects.
        ba_input_graph: GtsfmData input to bundle adjustment wrapped as Delayed.
        ba_output_graph: GtsfmData output to bundle adjustment wrapped as Delayed.
        results_path: Path to directory where GTSFM results will be saved.

    Returns:
        A list of delayed objects after saving the input and outputs to bundle adjustment.
    """
    start_time = time.time()
    saving_graph_list = []
    
    output_dir = results_path
    # Save the input to Bundle Adjustment (from data association).
    saving_graph_list.append(
        dask.delayed(io_utils.export_model_as_colmap_text)(
            gtsfm_data=ba_input_graph,
            images=image_graph,
            save_dir=os.path.join(output_dir, "ba_input"),
        )
    )
    # Save the output of Bundle Adjustment.
    saving_graph_list.append(
        dask.delayed(io_utils.export_model_as_colmap_text)(
            gtsfm_data=ba_output_graph,
            images=image_graph,
            save_dir=os.path.join(output_dir, "ba_output"),
        )
    )

    # Save the ground truth in the same format, for visualization.
    # We use the estimated tracks here, with ground truth camera poses.
    gt_gtsfm_data = dask.delayed(get_gtsfm_data_with_gt_cameras_and_est_tracks)(cameras_gt, ba_output_graph)
    saving_graph_list.append(
        dask.delayed(io_utils.export_model_as_colmap_text)(
            gtsfm_data=gt_gtsfm_data,
            images=image_graph,
            save_dir=os.path.join(output_dir, "ba_output_gt"),
        )
    )

    # Save a duplicate in REACT_RESULTS_PATH.
    end_time = time.time()
    duration_sec = end_time - start_time
    logger.info("GtsfmData I/O took %.2f sec.", duration_sec)
    shutil.copytree(src=results_path, dst=REACT_RESULTS_PATH)
    return saving_graph_list


def save_full_frontend_metrics(
    two_view_report_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    images: List[Image],
    filename: str,
    metrics_path: Path,
    plot_base_path: Path,
    save_retrieval_metrics: bool = True,
) -> None:
    """Converts the TwoViewEstimationReports for all image pairs to a Dict and saves it as JSON.

    Args:
        two_view_report_dict: Front-end metrics for pairs of images.
        images: List of all images for this scene, in order of image/frame index.
        filename: File name to use when saving report to JSON.
        matching_regime: Regime used for image pair selection in retriever.
        metrics_path: Path to directory where metrics will be saved.
        plot_base_path: Path to directory where plots will be saved.
    """
    metrics_list = two_view_estimator.get_two_view_reports_summary(two_view_report_dict, images)

    io_utils.save_json_file(os.path.join(metrics_path, filename), metrics_list)

    # Save duplicate copy of 'frontend_full.json' within React Folder.
    io_utils.save_json_file(os.path.join(REACT_METRICS_PATH, filename), metrics_list)

    # All retreival metrics need GT, no need to save them if GT is not available.
    gt_available = any([report.R_error_deg is not None for report in two_view_report_dict.values()])

    if save_retrieval_metrics and "VIEWGRAPH_2VIEW_REPORT" in filename and gt_available:
        # must come after two-view report file is written to disk in the Dask dependency graph.
        _save_retrieval_two_view_metrics(metrics_path, plot_base_path)


def _save_retrieval_two_view_metrics(metrics_path: Path, plot_base_path: Path) -> None:
    """Compare 2-view similarity scores with their 2-view pose errors after viewgraph estimation."""
    sim_fpath = plot_base_path / "netvlad_similarity_matrix.txt"
    if not sim_fpath.exists():
        logger.warning(msg="NetVLAD similarity matrix not found. Skipping retrieval metrics.")
        return

    sim = np.loadtxt(str(sim_fpath), delimiter=",")
    json_data = io_utils.read_json_file(metrics_path / "two_view_report_VIEWGRAPH_2VIEW_REPORT.json")

    sim_scores = []
    R_errors = []
    U_errors = []

    for entry in json_data:
        i1 = entry["i1"]
        i2 = entry["i2"]
        R_error = entry["rotation_angular_error"]
        U_error = entry["translation_angular_error"]
        if R_error is None or U_error is None:
            continue
        sim_score = sim[i1, i2]

        sim_scores.append(sim_score)
        R_errors.append(R_error)
        U_errors.append(U_error)

    plt.scatter(sim_scores, R_errors, 10, color="r", marker=".")
    plt.xlabel("Similarity score")
    plt.ylabel("Rotation error w.r.t. GT (deg.)")
    plt.savefig(os.path.join(plot_base_path, "gt_rot_error_vs_similarity_score.jpg"), dpi=500)
    plt.close("all")

    plt.scatter(sim_scores, U_errors, 10, color="r", marker=".")
    plt.xlabel("Similarity score")
    plt.ylabel("Translation direction error w.r.t. GT (deg.)")
    plt.savefig(os.path.join(plot_base_path, "gt_trans_error_vs_similarity_score.jpg"), dpi=500)
    plt.close("all")

    pose_errors = np.maximum(np.array(R_errors), np.array(U_errors))
    plt.scatter(sim_scores, pose_errors, 10, color="r", marker=".")
    plt.xlabel("Similarity score")
    plt.ylabel("Pose error w.r.t. GT (deg.)")
    plt.savefig(os.path.join(plot_base_path, "gt_pose_error_vs_similarity_score.jpg"), dpi=500)
    plt.close("all")
