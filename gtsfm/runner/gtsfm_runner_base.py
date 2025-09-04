"""Base class for runner that executes SfM."""

import argparse
import os
import time
from abc import abstractmethod, abstractproperty
from pathlib import Path
import logging

import dask
import hydra
import numpy as np
from dask import config as dask_config
from dask.distributed import Client, LocalCluster, SSHCluster, performance_report
from gtsam import Pose3, Rot3, Unit3
from hydra.utils import instantiate
from omegaconf import OmegaConf

import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.merging as merging_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.viz as viz_utils
from gtsfm import two_view_estimator
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import ImageMatchingRegime
from gtsfm.scene_optimizer import SceneOptimizer
from gtsfm.two_view_estimator import TWO_VIEW_OUTPUT, TwoViewEstimationReport, run_two_view_estimator_as_futures
from gtsfm.ui.process_graph_generator import ProcessGraphGenerator
from gtsfm.utils.subgraph_utils import group_results_by_subgraph

dask_config.set({"distributed.scheduler.worker-ttl": None})

logger = logger_utils.get_logger()

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent
REACT_METRICS_PATH = DEFAULT_OUTPUT_ROOT / "rtf_vis_tool" / "src" / "result_metrics"


class GtsfmRunnerBase:
    @abstractproperty
    def tag(self):
        pass

    def __init__(self, override_args=None) -> None:
        argparser: argparse.ArgumentParser = self.construct_argparser()
        self.parsed_args: argparse.Namespace = argparser.parse_args(args=override_args)
        if self.parsed_args.dask_tmpdir:
            dask.config.set({"temporary_directory": DEFAULT_OUTPUT_ROOT / self.parsed_args.dask_tmpdir})

        # Get the numeric level from the string
        log_level = getattr(logging, self.parsed_args.log.upper(), None)

        # 5. Configure the logging system
        # A good format includes the timestamp, level name, and message
        logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s")

        self.loader: LoaderBase = self.construct_loader()
        self.scene_optimizer: SceneOptimizer = self.construct_scene_optimizer()
        self.graph_partitioner: GraphPartitionerBase = self.scene_optimizer.graph_partitioner

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=self.tag)

        parser.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="Number of workers to start (processes, by default).",
        )
        parser.add_argument(
            "--threads_per_worker",
            type=int,
            default=1,
            help="Number of threads per each worker.",
        )
        parser.add_argument(
            "--worker_memory_limit", type=str, default="8GB", help="Memory limit per worker, e.g. `8GB`"
        )
        parser.add_argument(
            "--config_name",
            type=str,
            default="sift_front_end.yaml",
            help="Master config, including back-end configuration. Options include `unified_config.yaml`,"
            " `sift_front_end.yaml`, `deep_front_end.yaml`, etc.",
        )
        parser.add_argument(
            "--correspondence_generator_config_name",
            type=str,
            default=None,
            help="Override flag for correspondence generator (choose from among gtsfm/configs/correspondence).",
        )
        parser.add_argument(
            "--verifier_config_name",
            type=str,
            default=None,
            help="Override flag for verifier (choose from among gtsfm/configs/verifier).",
        )
        parser.add_argument(
            "--retriever_config_name",
            type=str,
            default=None,
            help="Override flag for retriever (choose from among gtsfm/configs/retriever).",
        )
        parser.add_argument(
            "--gaussian_splatting_config_name",
            type=str,
            default="base_gs",
            help="Override flag for your own gaussian splatting implementation.",
        )
        parser.add_argument(
            "--max_resolution",
            type=int,
            default=760,
            help="integer representing maximum length of image's short side"
            " e.g. for 1080p (1920 x 1080), max_resolution would be 1080",
        )
        parser.add_argument(
            "--max_frame_lookahead",
            type=int,
            default=None,
            help="Maximum number of consecutive frames to consider for matching/co-visibility.",
        )
        parser.add_argument(
            "--num_matched",
            type=int,
            default=None,
            help="Number of K potential matches to provide per query. These are the top `K` matches per query.",
        )
        parser.add_argument(
            "--share_intrinsics", action="store_true", help="Shares the intrinsics between all the cameras."
        )
        parser.add_argument("--run_mvs", action="store_true", help="Run dense MVS reconstruction")
        parser.add_argument("--run_gs", action="store_true", help="Run Gaussian Splatting")
        parser.add_argument(
            "--output_root",
            type=str,
            default=DEFAULT_OUTPUT_ROOT,
            help="Root directory. Results, plots and metrics will be stored in subdirectories,"
            " e.g. {output_root}/results",
        )
        parser.add_argument(
            "--dask_tmpdir",
            type=str,
            default=None,
            help="tmp directory for dask workers, uses dask's default (/tmp) if not set",
        )
        parser.add_argument(
            "--cluster_config",
            type=str,
            default=None,
            help="config listing IP worker addresses for the cluster,"
            " first worker is used as scheduler and should contain the dataset",
        )
        parser.add_argument(
            "--dashboard_port",
            type=str,
            default=":8787",
            help="dask dashboard port number",
        )
        parser.add_argument(
            "--num_retry_cluster_connection",
            type=int,
            default=3,
            help="Number of times to retry cluster connection if it fails.",
        )
        parser.add_argument(
            "--graph_partitioner",
            type=str,
            default="single",
            choices=["single", "other_partitioner_types"],
            help="Type of graph partitioner to use. Default is 'single' (SinglePartition).",
        )
        parser.add_argument(
            "-l",
            "--log",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",  # Set a default level
            help="Set the logging level",
        )
        return parser

    @abstractmethod
    def construct_loader(self) -> LoaderBase:
        pass

    def construct_scene_optimizer(self) -> SceneOptimizer:
        """Construct scene optimizer.

        All configs are relative to the gtsfm module.
        """
        with hydra.initialize_config_module(config_module="gtsfm.configs"):
            overrides = ["+SceneOptimizer.output_root=" + str(self.parsed_args.output_root)]
            if self.parsed_args.share_intrinsics:
                overrides.append("SceneOptimizer.multiview_optimizer.bundle_adjustment_module.shared_calib=True")

            main_cfg = hydra.compose(
                config_name=self.parsed_args.config_name,
                overrides=overrides,
            )
            scene_optimizer: SceneOptimizer = instantiate(main_cfg.SceneOptimizer)

        # Override correspondence generator.
        if self.parsed_args.correspondence_generator_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.correspondence"):
                correspondence_cfg = hydra.compose(
                    config_name=self.parsed_args.correspondence_generator_config_name,
                )
                logger.info("\n\nCorrespondenceGenerator override: " + OmegaConf.to_yaml(correspondence_cfg))
                scene_optimizer.correspondence_generator = instantiate(correspondence_cfg.CorrespondenceGenerator)

        # Override verifier.
        if self.parsed_args.verifier_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.verifier"):
                verifier_cfg = hydra.compose(
                    config_name=self.parsed_args.verifier_config_name,
                )
                logger.info("\n\nVerifier override: " + OmegaConf.to_yaml(verifier_cfg))
                scene_optimizer.two_view_estimator._verifier = instantiate(verifier_cfg.verifier)

        # Override retriever.
        if self.parsed_args.retriever_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.retriever"):
                retriever_cfg = hydra.compose(
                    config_name=self.parsed_args.retriever_config_name,
                )
                logger.info("\n\nRetriever override: " + OmegaConf.to_yaml(retriever_cfg))
                scene_optimizer.image_pairs_generator._retriever = instantiate(retriever_cfg.retriever)

        # Override gaussian splatting
        if self.parsed_args.gaussian_splatting_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.gaussian_splatting"):
                gs_cfg = hydra.compose(
                    config_name=self.parsed_args.gaussian_splatting_config_name,
                )
                logger.info("\n\nGaussian Splatting override: " + OmegaConf.to_yaml(gs_cfg))
                scene_optimizer.gaussian_splatting_optimizer = instantiate(gs_cfg.gaussian_splatting_optimizer)

        if self.parsed_args.max_frame_lookahead is not None:
            if scene_optimizer.image_pairs_generator._retriever._matching_regime in [
                ImageMatchingRegime.SEQUENTIAL,
                ImageMatchingRegime.SEQUENTIAL_HILTI,
            ]:
                scene_optimizer.image_pairs_generator._retriever._max_frame_lookahead = (
                    self.parsed_args.max_frame_lookahead
                )
            elif (
                scene_optimizer.image_pairs_generator._retriever._matching_regime
                == ImageMatchingRegime.SEQUENTIAL_WITH_RETRIEVAL
            ):
                scene_optimizer.image_pairs_generator._retriever._seq_retriever._max_frame_lookahead = (
                    self.parsed_args.max_frame_lookahead
                )
            else:
                raise ValueError(
                    "`max_frame_lookahead` arg is incompatible with retriever matching regime "
                    f"{scene_optimizer.image_pairs_generator._retriever._matching_regime}"
                )
        if self.parsed_args.num_matched is not None:
            if (
                scene_optimizer.image_pairs_generator._retriever._matching_regime
                == ImageMatchingRegime.SEQUENTIAL_WITH_RETRIEVAL
            ):
                scene_optimizer.image_pairs_generator._retriever._similarity_retriever._num_matched = (
                    self.parsed_args.num_matched
                )
            elif scene_optimizer.image_pairs_generator._retriever._matching_regime == ImageMatchingRegime.RETRIEVAL:
                scene_optimizer.image_pairs_generator._retriever._num_matched = self.parsed_args.num_matched
            else:
                raise ValueError(
                    "`num_matched` arg is incompatible with retriever matching regime "
                    f"{scene_optimizer.image_pairs_generator._retriever._matching_regime}"
                )

        if not self.parsed_args.run_mvs:
            scene_optimizer.run_dense_optimizer = False

        if not self.parsed_args.run_gs:
            scene_optimizer.run_gaussian_splatting_optimizer = False

        logger.info("\n\nSceneOptimizer: " + str(scene_optimizer))
        return scene_optimizer

    def setup_ssh_cluster_with_retries(self) -> SSHCluster:
        """Sets up SSH Cluster allowing multiple retries upon connection failures."""
        workers = OmegaConf.load(os.path.join("gtsfm", "configs", self.parsed_args.cluster_config))["workers"]
        scheduler = workers[0]
        connected = False
        retry_count = 0
        while retry_count < self.parsed_args.num_retry_cluster_connection and not connected:
            logger.info(f"Connecting to the cluster: attempt {retry_count + 1}")
            logger.info(f"Using {scheduler} as scheduler")
            logger.info(f"Using {workers} as workers")
            try:
                cluster = SSHCluster(
                    [scheduler] + workers,
                    scheduler_options={"dashboard_address": self.parsed_args.dashboard_port},
                    worker_options={
                        "n_workers": self.parsed_args.num_workers,
                        "nthreads": self.parsed_args.threads_per_worker,
                        "memory_limit": self.parsed_args.worker_memory_limit,
                    },
                )
                connected = True
            except Exception as e:
                logger.info(f"Worker failed to start: {str(e)}")
                retry_count += 1
        if not connected:
            raise ValueError(
                f"Connection to cluster could not be established after {self.parsed_args.num_retry_cluster_connection}"
                " attempts. Aborting..."
            )
        return cluster

    def run(self) -> GtsfmData:
        """Run the SceneOptimizer."""
        start_time = time.time()

        # Create dask cluster.
        if self.parsed_args.cluster_config:
            cluster = self.setup_ssh_cluster_with_retries()
            client = Client(cluster)
            # getting first worker's IP address and port to do IO
            io_worker = list(client.scheduler_info()["workers"].keys())[0]
            self.loader._input_worker = io_worker
            self.scene_optimizer._output_worker = io_worker
        else:
            local_cluster_kwargs = {
                "n_workers": self.parsed_args.num_workers,
                "threads_per_worker": self.parsed_args.threads_per_worker,
                "dashboard_address": self.parsed_args.dashboard_port,
            }
            if self.parsed_args.worker_memory_limit is not None:
                local_cluster_kwargs["memory_limit"] = self.parsed_args.worker_memory_limit
            cluster = LocalCluster(**local_cluster_kwargs)
            client = Client(cluster)

        # Display Dask dashboard URL before processing starts
        print(f"\n🚀 Dask Dashboard available at: {client.dashboard_link}")

        # Create process graph.
        process_graph_generator = ProcessGraphGenerator()
        if isinstance(self.scene_optimizer.correspondence_generator, ImageCorrespondenceGenerator):
            process_graph_generator.is_image_correspondence = True
        process_graph_generator.save_graph()

        retriever_start_time = time.time()
        with performance_report(filename="retriever-dask-report.html"):
            image_pair_indices = self.scene_optimizer.image_pairs_generator.generate_image_pairs(
                client=client,
                images=self.loader.get_all_images_as_futures(client),
                image_fnames=self.loader.image_filenames(),
                plots_output_dir=self.scene_optimizer.create_plot_base_path(),
            )

        retriever_metrics = self.scene_optimizer.image_pairs_generator._retriever.evaluate(
            len(self.loader), image_pair_indices
        )
        retriever_duration_sec = time.time() - retriever_start_time
        retriever_metrics.add_metric(GtsfmMetric("retriever_duration_sec", retriever_duration_sec))
        logger.info("Image pair retrieval took %.2f sec.", retriever_duration_sec)

        intrinsics = self.loader.get_all_intrinsics()

        with performance_report(filename="correspondence-generator-dask-report.html"):
            correspondence_generation_start_time = time.time()
            (
                keypoints_list,
                putative_corr_idxs_dict,
            ) = self.scene_optimizer.correspondence_generator.generate_correspondences(
                client,
                self.loader.get_all_images_as_futures(client),
                image_pair_indices,
            )
            correspondence_generation_duration_sec = time.time() - correspondence_generation_start_time

            two_view_estimation_start_time = time.time()
            two_view_results_dict = run_two_view_estimator_as_futures(
                client,
                self.scene_optimizer.two_view_estimator,
                keypoints_list,
                putative_corr_idxs_dict,
                intrinsics,
                self.loader.get_relative_pose_priors(image_pair_indices),
                self.loader.get_gt_cameras(),
                gt_scene_mesh=self.loader.get_gt_scene_trimesh(),
            )
            two_view_estimation_duration_sec = time.time() - two_view_estimation_start_time

        i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, pre_ba_two_view_reports_dict, post_isp_two_view_reports_dict = (
            unzip_two_view_results(two_view_results_dict)
        )

        if self.scene_optimizer._save_two_view_correspondences_viz:
            for i1, i2 in v_corr_idxs_dict.keys():
                image_i1 = self.loader.get_image(i1)
                image_i2 = self.loader.get_image(i2)
                viz_utils.save_twoview_correspondences_viz(
                    image_i1,
                    image_i2,
                    keypoints_list[i1],
                    keypoints_list[i2],
                    v_corr_idxs_dict[(i1, i2)],
                    two_view_report=post_isp_two_view_reports_dict[(i1, i2)],
                    file_path=os.path.join(
                        self.scene_optimizer._plot_correspondence_path,
                        f"{i1}_{i2}__{image_i1.file_name}_{image_i2.file_name}.jpg",
                    ),
                )

        two_view_agg_metrics = two_view_estimator.aggregate_frontend_metrics(
            two_view_reports_dict=post_isp_two_view_reports_dict,
            angular_err_threshold_deg=self.scene_optimizer._pose_angular_error_thresh,
            metric_group_name="verifier_summary_{}".format(two_view_estimator.POST_ISP_REPORT_TAG),
        )
        two_view_agg_metrics.add_metric(
            GtsfmMetric("total_correspondence_generation_duration_sec", correspondence_generation_duration_sec)
        )
        two_view_agg_metrics.add_metric(
            GtsfmMetric("total_two_view_estimation_duration_sec", two_view_estimation_duration_sec)
        )
        all_metrics_groups = [retriever_metrics, two_view_agg_metrics]

        # Partition image pairs
        subgraphs = self.graph_partitioner.partition_image_pairs(image_pair_indices)
        logger.info(f"Partitioned into {len(subgraphs)} subgraphs")
        # Group results by subgraph
        subgraph_two_view_results = group_results_by_subgraph(two_view_results_dict, subgraphs)

        # Process each subgraph
        all_delayed_sfm_results = []
        all_delayed_io = []
        all_delayed_mvo_metrics_groups = []

        for idx, subgraph_result_dict in enumerate(subgraph_two_view_results):
            logger.info(
                f"Creating computation graph for subgraph {idx + 1}/{len(subgraph_two_view_results)} "
                f"with {    len(subgraph_result_dict)} image pairs"
            )
            if len(subgraph_two_view_results) == 1:
                # single partition
                self.scene_optimizer.create_output_directories(None)
            else:
                self.scene_optimizer.create_output_directories(idx + 1)

            # Unzip the two-view results for this subgraph
            subgraph_i2Ri1_dict, subgraph_i2Ui1_dict, subgraph_v_corr_idxs_dict, _, subgraph_post_isp_reports = (
                unzip_two_view_results(subgraph_result_dict)
            )

            # Create computation graph for this subgraph
            if len(subgraph_i2Ri1_dict) > 0:  # Only process non-empty subgraphs
                delayed_sfm_result, delayed_io, delayed_mvo_metrics_groups = (
                    self.scene_optimizer.create_computation_graph(
                        keypoints_list=keypoints_list,
                        i2Ri1_dict=subgraph_i2Ri1_dict,
                        i2Ui1_dict=subgraph_i2Ui1_dict,
                        v_corr_idxs_dict=subgraph_v_corr_idxs_dict,
                        two_view_reports=subgraph_post_isp_reports,
                        num_images=len(self.loader),
                        images=self.loader.create_computation_graph_for_images(),
                        camera_intrinsics=intrinsics,
                        relative_pose_priors=self.loader.get_relative_pose_priors(list(subgraph_i2Ri1_dict.keys())),
                        absolute_pose_priors=self.loader.get_absolute_pose_priors(),
                        cameras_gt=self.loader.get_gt_cameras(),
                        gt_wTi_list=self.loader.get_gt_poses(),
                        gt_scene_mesh=self.loader.get_gt_scene_trimesh(),
                    )
                )
                all_delayed_sfm_results.append(delayed_sfm_result)
                all_delayed_io.extend(delayed_io)
                all_delayed_mvo_metrics_groups.extend(delayed_mvo_metrics_groups)
            else:
                logger.warning(f"Skipping subgraph {idx+1} as it has no valid two-view results.")

        # Compute all the delayed objects
        with performance_report(filename="scene-optimizer-dask-report.html"):
            if all_delayed_sfm_results:
                results = dask.compute(*all_delayed_sfm_results, *all_delayed_io, *all_delayed_mvo_metrics_groups)
                sfm_results = results[: len(all_delayed_sfm_results)]
                other_results = results[len(all_delayed_sfm_results) :]

                # Extract metrics from results
                mvo_metrics_groups = [x for x in other_results if isinstance(x, GtsfmMetricsGroup)]
                all_metrics_groups.extend(mvo_metrics_groups)

                # For now, return the first non-empty result
                sfm_result = next((r for r in sfm_results if r is not None), None)
            else:
                sfm_result = None

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)

        if client is not None:
            client.shutdown()

        # Add total summary metrics
        total_summary_metrics = GtsfmMetricsGroup(
            "total_summary_metrics", [GtsfmMetric("total_runtime_sec", duration_sec)]
        )
        all_metrics_groups.append(total_summary_metrics)

        # Save metrics reports
        save_metrics_reports(all_metrics_groups, os.path.join(self.scene_optimizer.output_root, "result_metrics"))

        return sfm_result


def unzip_two_view_results(two_view_results: dict[tuple[int, int], TWO_VIEW_OUTPUT]) -> tuple[
    dict[tuple[int, int], Rot3],
    dict[tuple[int, int], Unit3],
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], TwoViewEstimationReport],
    dict[tuple[int, int], TwoViewEstimationReport],
]:
    """Unzip the tuple TWO_VIEW_OUTPUT into 1 dictionary for 1 element in the tuple."""
    i2Ri1_dict: dict[tuple[int, int], Rot3] = {}
    i2Ui1_dict: dict[tuple[int, int], Unit3] = {}
    v_corr_idxs_dict: dict[tuple[int, int], np.ndarray] = {}
    pre_ba_two_view_reports_dict: dict[tuple[int, int], TwoViewEstimationReport] = {}
    post_isp_two_view_reports_dict: dict[tuple[int, int], TwoViewEstimationReport] = {}

    for (i1, i2), two_view_output in two_view_results.items():
        # Value is ordered as (post_isp_i2Ri1, post_isp_i2Ui1, post_isp_v_corr_idxs,
        # pre_ba_report, post_ba_report, post_isp_report).
        i2Ri1 = two_view_output[0]
        i2Ui1 = two_view_output[1]
        if i2Ri1 is None or i2Ui1 is None:
            logger.debug("Skip %d, %d since None", i1, i2)
            continue

        i2Ri1_dict[(i1, i2)] = i2Ri1
        i2Ui1_dict[(i1, i2)] = i2Ui1
        v_corr_idxs_dict[(i1, i2)] = two_view_output[2]
        pre_ba_two_view_reports_dict[(i1, i2)] = two_view_output[3]
        post_isp_two_view_reports_dict[(i1, i2)] = two_view_output[5]

    return i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, pre_ba_two_view_reports_dict, post_isp_two_view_reports_dict


def save_metrics_reports(metrics_group_list: list[GtsfmMetricsGroup], metrics_path: str) -> None:
    """Saves metrics to JSON and HTML report.

    Args:
        metrics_graph: list of GtsfmMetricsGroup from different modules wrapped as Delayed.
        metrics_path: Path to directory where computed metrics will be saved.
    """

    # Save metrics to JSON
    metrics_utils.save_metrics_as_json(metrics_group_list, metrics_path)
    metrics_utils.save_metrics_as_json(metrics_group_list, str(REACT_METRICS_PATH))

    metrics_report.generate_metrics_report_html(
        metrics_group_list, os.path.join(metrics_path, "gtsfm_metrics_report.html"), None
    )


def merge_two_partition_results(poses1: dict[int, Pose3], poses2: dict[int, Pose3]) -> dict[int, Pose3]:
    """
    Merges poses from two partitions by finding and applying relative transform aTb.

    Assumes poses1 are relative to frame 'a' and poses2 are relative to frame 'b'.
    Finds 'aTb' (from frame 'b' to frame 'a') via overlapping poses.
    Transforms non-overlapping poses from partition 2 into frame 'a' and merges.

    Args:
        poses1: dictionary {camera_index: pose_in_frame_a}.
        poses2: dictionary {camera_index: pose_in_frame_b}.

    Returns:
        A merged dictionary {camera_index: pose_in_frame_a}.

    Raises:
        ValueError: If no overlapping cameras are found between the two partitions.
        RuntimeError: If GTSAM optimization fails.
    """
    keys, pairs = merging_utils._get_overlap_data(poses1, poses2)
    aTb = merging_utils._calculate_transform(pairs)
    return merging_utils._merge_poses_final(poses1, poses2, keys, aTb)
