"""Base class for runner that executes SfM."""

import argparse
import logging
import os
import time
from pathlib import Path

import dask
import dask.config
import hydra
from dask import config as dask_config
from dask.distributed import Client, LocalCluster, SSHCluster, performance_report
from gtsam import Pose3  # type: ignore
from hydra.utils import instantiate
from omegaconf import OmegaConf

import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.merging as merging_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.viz as viz_utils
from gtsfm import two_view_estimator
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.types import CALIBRATION_TYPE
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.scene_optimizer import SceneOptimizer
from gtsfm.two_view_estimator import run_two_view_estimator_as_futures
from gtsfm.ui.process_graph_generator import ProcessGraphGenerator
from gtsfm.utils.configuration import log_configuration_summary, log_full_configuration, log_key_parameters
from gtsfm.utils.subgraph_utils import group_results_by_subgraph

dask_config.set({"distributed.scheduler.worker-ttl": None})

logger = logger_utils.get_logger()

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent
REACT_METRICS_PATH = DEFAULT_OUTPUT_ROOT / "rtf_vis_tool" / "src" / "result_metrics"


class GtsfmRunner:
    @property
    def tag(self) -> str:
        return "Unified GTSFM Runner"

    def __init__(self, override_args=None) -> None:
        argparser: argparse.ArgumentParser = self.construct_argparser()
        self.parsed_args: argparse.Namespace = argparser.parse_args(args=override_args)
        if self.parsed_args.dask_tmpdir:
            dask.config.set({"temporary_directory": DEFAULT_OUTPUT_ROOT / self.parsed_args.dask_tmpdir})

        # Configure the logging system
        log_level = getattr(logging, self.parsed_args.log.upper(), None)
        if log_level is not None:
            logger.setLevel(log_level)

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

        # Loader configuration
        parser.add_argument(
            "--loader",
            type=str,
            default="olsson_loader",
            help="Loader type. Available options: colmap_loader, hilti_loader, astrovision_loader, "
            "olsson_loader, argoverse_loader, mobilebrick_loader, one_d_sfm_loader, "
            "tanks_and_temples_loader, yfcc_imb_loader. Default: olsson_loader",
        )

        # Standardized loader arguments
        parser.add_argument("--dataset_dir", type=str, help="Path to dataset directory")
        parser.add_argument(
            "--images_dir", type=str, help="Path to images directory (optional, defaults depend on loader)"
        )
        parser.add_argument("--max_length", type=int, help="Maximum number of images/timestamps to process")
        parser.add_argument("--scene_name", type=str, help="Name of the scene (for Tanks and Temples, etc.)")

        # Argoverse-specific arguments
        parser.add_argument("--log_id", type=str, help="Unique ID of vehicle log (Argoverse)")
        parser.add_argument("--stride", type=int, help="Sampling rate, e.g. every N images (Argoverse)")
        parser.add_argument("--max_num_imgs", type=int, help="Maximum number of images to load (Argoverse)")
        parser.add_argument("--max_lookahead_sec", type=float, help="Maximum lookahead in seconds (Argoverse)")
        parser.add_argument("--camera_name", type=str, help="Camera name to use (Argoverse)")

        # MobileBrick/COLMAP-specific arguments
        parser.add_argument(
            "--use_gt_intrinsics", action="store_true", help="Use ground truth intrinsics (MobileBrick/COLMAP)"
        )
        parser.add_argument("--use_gt_extrinsics", action="store_true", help="Use ground truth extrinsics (COLMAP)")

        # 1DSFM-specific arguments
        parser.add_argument("--enable_no_exif", action="store_true", help="Read images without EXIF (1DSFM)")
        parser.add_argument("--default_focal_length_factor", type=float, help="Default focal length factor (1DSFM)")

        # Tanks and Temples-specific arguments
        parser.add_argument("--poses_fpath", type=str, help="Path to poses file (Tanks and Temples)")
        parser.add_argument(
            "--bounding_polyhedron_json_fpath",
            type=str,
            help="Path to bounding polyhedron JSON (Tanks and Temples)",
        )
        parser.add_argument("--ply_alignment_fpath", type=str, help="Path to PLY alignment file (Tanks and Temples)")
        parser.add_argument("--lidar_ply_fpath", type=str, help="Path to LiDAR PLY file (Tanks and Temples)")
        parser.add_argument("--colmap_ply_fpath", type=str, help="Path to COLMAP PLY file (Tanks and Temples)")
        parser.add_argument("--max_num_images", type=int, help="Maximum number of images (Tanks and Temples)")

        # YFCC IMB-specific arguments
        parser.add_argument("--co_visibility_threshold", type=float, help="Co-visibility threshold (YFCC IMB)")

        return parser

    def construct_scene_optimizer(self) -> SceneOptimizer:
        """Construct scene optimizer.

        All configs are relative to the gtsfm module.
        """
        logger.info(f"📁 Config File: {self.parsed_args.config_name}")
        with hydra.initialize_config_module(config_module="gtsfm.configs", version_base=None):
            overrides = ["+SceneOptimizer.output_root=" + str(self.parsed_args.output_root)]
            if self.parsed_args.share_intrinsics:
                overrides.append("SceneOptimizer.multiview_optimizer.bundle_adjustment_module.shared_calib=True")

            # Add loader overrides based on command line arguments
            if self.parsed_args.loader:
                overrides.append("~loader")
                overrides.append(f"+loader={self.parsed_args.loader}")

            # Standardized loader parameter overrides
            if self.parsed_args.dataset_dir:
                overrides.append(f"loader.dataset_dir={self.parsed_args.dataset_dir}")
            if self.parsed_args.images_dir:
                overrides.append(f"loader.images_dir={self.parsed_args.images_dir}")

            # Loader-specific parameter overrides
            if self.parsed_args.max_length is not None:
                overrides.append(f"loader.max_length={self.parsed_args.max_length}")

            # Argoverse-specific overrides
            if self.parsed_args.log_id and self.parsed_args.loader == "argoverse_loader":
                overrides.append(f"loader.log_id={self.parsed_args.log_id}")
            if self.parsed_args.stride is not None and self.parsed_args.loader == "argoverse_loader":
                overrides.append(f"loader.stride={self.parsed_args.stride}")
            if self.parsed_args.max_num_imgs is not None and self.parsed_args.loader == "argoverse_loader":
                overrides.append(f"loader.max_num_imgs={self.parsed_args.max_num_imgs}")
            if self.parsed_args.max_lookahead_sec is not None and self.parsed_args.loader == "argoverse_loader":
                overrides.append(f"loader.max_lookahead_sec={self.parsed_args.max_lookahead_sec}")
            if self.parsed_args.camera_name and self.parsed_args.loader == "argoverse_loader":
                overrides.append(f"loader.camera_name={self.parsed_args.camera_name}")

            # MobileBrick/COLMAP-specific overrides
            if self.parsed_args.use_gt_intrinsics and self.parsed_args.loader in [
                "mobilebrick_loader",
                "colmap_loader",
            ]:
                overrides.append(f"loader.use_gt_intrinsics={self.parsed_args.use_gt_intrinsics}")
            if self.parsed_args.use_gt_extrinsics and self.parsed_args.loader == "colmap_loader":
                overrides.append(f"loader.use_gt_extrinsics={self.parsed_args.use_gt_extrinsics}")

            # Argoverse-specific overrides that support max_frame_lookahead
            if self.parsed_args.max_frame_lookahead is not None and self.parsed_args.loader in [
                "argoverse_loader",
                "hilti_loader",
            ]:
                overrides.append(f"loader.max_frame_lookahead={self.parsed_args.max_frame_lookahead}")

            # 1DSFM-specific overrides
            if self.parsed_args.enable_no_exif and self.parsed_args.loader == "one_d_sfm_loader":
                overrides.append(f"loader.enable_no_exif={self.parsed_args.enable_no_exif}")
            if (
                self.parsed_args.default_focal_length_factor is not None
                and self.parsed_args.loader == "one_d_sfm_loader"
            ):
                overrides.append(f"loader.default_focal_length_factor={self.parsed_args.default_focal_length_factor}")

            # Tanks and Temples-specific overrides
            if self.parsed_args.poses_fpath and self.parsed_args.loader == "tanks_and_temples_loader":
                overrides.append(f"loader.poses_fpath={self.parsed_args.poses_fpath}")
            if (
                self.parsed_args.bounding_polyhedron_json_fpath
                and self.parsed_args.loader == "tanks_and_temples_loader"
            ):
                overrides.append(
                    f"loader.bounding_polyhedron_json_fpath={self.parsed_args.bounding_polyhedron_json_fpath}"
                )
            if self.parsed_args.ply_alignment_fpath and self.parsed_args.loader == "tanks_and_temples_loader":
                overrides.append(f"loader.ply_alignment_fpath={self.parsed_args.ply_alignment_fpath}")
            if self.parsed_args.lidar_ply_fpath and self.parsed_args.loader == "tanks_and_temples_loader":
                overrides.append(f"loader.lidar_ply_fpath={self.parsed_args.lidar_ply_fpath}")
            if self.parsed_args.colmap_ply_fpath and self.parsed_args.loader == "tanks_and_temples_loader":
                overrides.append(f"loader.colmap_ply_fpath={self.parsed_args.colmap_ply_fpath}")
            if self.parsed_args.max_num_images is not None and self.parsed_args.loader == "tanks_and_temples_loader":
                overrides.append(f"loader.max_num_images={self.parsed_args.max_num_images}")

            # YFCC IMB-specific overrides
            if self.parsed_args.co_visibility_threshold is not None and self.parsed_args.loader == "yfcc_imb_loader":
                overrides.append(f"loader.co_visibility_threshold={self.parsed_args.co_visibility_threshold}")

            # Override max_resolution for loader if specified
            overrides.append(f"loader.max_resolution={self.parsed_args.max_resolution}")

            main_cfg = hydra.compose(
                config_name=self.parsed_args.config_name,
                overrides=overrides,
            )
            logger.info("⏳ Instantiating SceneOptimizer...")
            scene_optimizer: SceneOptimizer = instantiate(main_cfg.SceneOptimizer)

        # Override correspondence generator.
        if self.parsed_args.correspondence_generator_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.correspondence", version_base=None):
                correspondence_cfg = hydra.compose(
                    config_name=self.parsed_args.correspondence_generator_config_name,
                )
                logger.info(
                    f"🔄 Applying Correspondence Override: " f"{self.parsed_args.correspondence_generator_config_name}"
                )
                scene_optimizer.correspondence_generator = instantiate(correspondence_cfg.CorrespondenceGenerator)

        # Override verifier.
        if self.parsed_args.verifier_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.verifier", version_base=None):
                verifier_cfg = hydra.compose(
                    config_name=self.parsed_args.verifier_config_name,
                )
                logger.info(f"🔄 Applying Verifier Override: {self.parsed_args.verifier_config_name}")
                scene_optimizer.two_view_estimator._verifier = instantiate(verifier_cfg.verifier)

        # Override retriever.
        if self.parsed_args.retriever_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.retriever", version_base=None):
                retriever_cfg = hydra.compose(
                    config_name=self.parsed_args.retriever_config_name,
                )
                logger.info(f"🔄 Applying Retriever Override: {self.parsed_args.retriever_config_name}")
                scene_optimizer.image_pairs_generator._retriever = instantiate(retriever_cfg.retriever)

        # Override gaussian splatting
        if self.parsed_args.gaussian_splatting_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.gaussian_splatting", version_base=None):
                gs_cfg = hydra.compose(
                    config_name=self.parsed_args.gaussian_splatting_config_name,
                )
                logger.info(
                    f"🔄 Applying Gaussian Splatting Override: " f"{self.parsed_args.gaussian_splatting_config_name}"
                )
                scene_optimizer.gaussian_splatting_optimizer = instantiate(gs_cfg.gaussian_splatting_optimizer)

        # Set retriever specific params if specified with CLI.
        retriever = scene_optimizer.image_pairs_generator._retriever
        if self.parsed_args.max_frame_lookahead is not None:
            try:
                retriever.set_max_frame_lookahead(self.parsed_args.max_frame_lookahead)
            except Exception as e:
                logger.warning(f"Failed to set max_frame_lookahead: {e}")
        if self.parsed_args.num_matched is not None:
            try:
                retriever.set_num_matched(self.parsed_args.num_matched)
            except Exception as e:
                logger.warning(f"Failed to set num_matched: {e}")

        if not self.parsed_args.run_mvs:
            scene_optimizer.run_dense_optimizer = False

        if not self.parsed_args.run_gs:
            scene_optimizer.run_gaussian_splatting_optimizer = False

        log_configuration_summary(main_cfg, logger)
        log_key_parameters(main_cfg, logger)
        log_full_configuration(main_cfg, logger)
        return scene_optimizer

    def setup_ssh_cluster_with_retries(self):
        """Sets up SSH Cluster allowing multiple retries upon connection failures."""
        config = OmegaConf.load(os.path.join("gtsfm", "configs", self.parsed_args.cluster_config))
        workers = dict(config)["workers"]
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
        all_metrics_groups = []
        self._create_process_graph()

        # Create Dask client
        client = self._create_dask_client()

        logger.info("🔥 GTSFM: Running image pair retrieval...")
        retriever_metrics, visibility_graph = self._run_retriever(client)
        all_metrics_groups.append(retriever_metrics)

        logger.info("🔥 GTSFM: Running correspondence generation...")
        maybe_intrinsics, intrinsics = self._get_intrinsics_or_raise()
        keypoints, putative_corr_idxs_dict, correspondence_duration_sec = self._run_correspondence_generation(
            client, visibility_graph
        )

        logger.info("🔥 GTSFM: Running two-view estimation...")
        two_view_results, tve_duration_sec = self._run_two_view_estimation(
            client, visibility_graph, keypoints, putative_corr_idxs_dict, intrinsics
        )

        # Aggregate two-view metrics
        all_metrics_groups.append(
            self._aggregate_two_view_metrics(keypoints, two_view_results, correspondence_duration_sec, tve_duration_sec)
        )

        logger.info("🔥 GTSFM: Partitioning the view graph...")
        subgraph_two_view_results = self._partition_view_graph(visibility_graph, two_view_results)

        logger.info("🔥 GTSFM: Create back-end computation subgraphs...")
        all_delayed_sfm_results = []
        all_delayed_io = []
        all_delayed_mvo_metrics_groups = []
        for idx, subgraph_two_view_results in enumerate(subgraph_two_view_results):
            delayed_sfm_result, delayed_io, delayed_mvo_metrics_groups = self._process_subgraph(
                idx, subgraph_two_view_results, keypoints, maybe_intrinsics, len(subgraph_two_view_results)
            )
            if delayed_sfm_result is not None:
                all_delayed_sfm_results.append(delayed_sfm_result)
            all_delayed_io.extend(delayed_io)
            all_delayed_mvo_metrics_groups.extend(delayed_mvo_metrics_groups)

        logger.info("🔥 GTSFM: Starting distributed computation with Dask...")
        with performance_report(filename="dask_reports/scene-optimizer.html"):
            if all_delayed_sfm_results:
                results = dask.compute(*all_delayed_sfm_results, *all_delayed_io, *all_delayed_mvo_metrics_groups)
                sfm_results = results[: len(all_delayed_sfm_results)]
                other_results = results[len(all_delayed_sfm_results) :]
                mvo_metrics_groups = [x for x in other_results if isinstance(x, GtsfmMetricsGroup)]
                all_metrics_groups.extend(mvo_metrics_groups)
                sfm_result = next((r for r in sfm_results if r is not None), None)
            else:
                sfm_result = None

        # Log total time taken and save metrics report
        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("🔥 GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)
        total_summary_metrics = GtsfmMetricsGroup(
            "total_summary_metrics", [GtsfmMetric("total_runtime_sec", duration_sec)]
        )
        all_metrics_groups.append(total_summary_metrics)
        save_metrics_reports(all_metrics_groups, os.path.join(self.scene_optimizer.output_root, "result_metrics"))

        # Shutdown the Dask client
        if client is not None:
            client.shutdown()
        return sfm_result  # type: ignore

    def _create_dask_client(self):
        if self.parsed_args.cluster_config:
            cluster = self.setup_ssh_cluster_with_retries()
            client = Client(cluster)
            # getting first worker's IP address and port to do IO
            io_worker = list(client.scheduler_info()["workers"].keys())[0]
            self.scene_optimizer.loader._input_worker = io_worker
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
        return client

    def _create_process_graph(self):
        process_graph_generator = ProcessGraphGenerator()
        if isinstance(self.scene_optimizer.correspondence_generator, ImageCorrespondenceGenerator):
            process_graph_generator.is_image_correspondence = True
        process_graph_generator.save_graph()

    def _run_retriever(self, client):
        retriever_start_time = time.time()
        with performance_report(filename="dask_reports/retriever.html"):
            visibility_graph = self.scene_optimizer.image_pairs_generator.run(
                client=client,
                images=self.scene_optimizer.loader.get_all_images_as_futures(client),
                image_fnames=self.scene_optimizer.loader.image_filenames(),
                plots_output_dir=self.scene_optimizer.create_plot_base_path(),
            )
        retriever_metrics = self.scene_optimizer.image_pairs_generator._retriever.evaluate(
            len(self.scene_optimizer.loader), visibility_graph
        )
        retriever_duration_sec = time.time() - retriever_start_time
        retriever_metrics.add_metric(GtsfmMetric("retriever_duration_sec", retriever_duration_sec))
        logger.info("🚀 Image pair retrieval took %.2f min.", retriever_duration_sec / 60.0)
        return retriever_metrics, visibility_graph

    def _get_intrinsics_or_raise(self):
        maybe_intrinsics = self.scene_optimizer.loader.get_all_intrinsics()
        # Check if maybe_intrinsics has any None values
        if any(intrinsic is None for intrinsic in maybe_intrinsics):
            raise ValueError("Some intrinsics are None. Please ensure all intrinsics are provided.")

        # If all intrinsics are valid, cast them to the correct type
        intrinsics: list[CALIBRATION_TYPE] = maybe_intrinsics  # type: ignore
        return maybe_intrinsics, intrinsics

    def _run_correspondence_generation(self, client, visibility_graph):
        with performance_report(filename="dask_reports/correspondence-generator.html"):
            correspondence_generation_start_time = time.time()
            (
                keypoints_list,
                putative_corr_idxs_dict,
            ) = self.scene_optimizer.correspondence_generator.generate_correspondences(
                client,
                self.scene_optimizer.loader.get_all_images_as_futures(client),
                visibility_graph,
            )
            correspondence_generation_duration_sec = time.time() - correspondence_generation_start_time
        return keypoints_list, putative_corr_idxs_dict, correspondence_generation_duration_sec

    def _run_two_view_estimation(self, client, visibility_graph, keypoints_list, putative_corr_idxs_dict, intrinsics):
        with performance_report(filename="dask_reports/two-view-estimation.html"):
            two_view_estimation_start_time = time.time()
            # TODO(Frank):this pulls *all* results to one machine! We might not want this.
            all_two_view_results = run_two_view_estimator_as_futures(
                client,
                self.scene_optimizer.two_view_estimator,
                keypoints_list,
                putative_corr_idxs_dict,
                intrinsics,
                self.scene_optimizer.loader.get_relative_pose_priors(visibility_graph),
                self.scene_optimizer.loader.get_gt_cameras(),
                gt_scene_mesh=self.scene_optimizer.loader.get_gt_scene_trimesh(),
            )
            two_view_estimation_duration_sec = time.time() - two_view_estimation_start_time
        # TODO(Frank): We might not be able to do this in a distributed manner
        two_view_results = {edge: tvr for edge, tvr in all_two_view_results.items() if tvr.valid()}
        return two_view_results, two_view_estimation_duration_sec

    def _maybe_save_two_view_viz(self, keypoints_list, two_view_results):
        if self.scene_optimizer._save_two_view_correspondences_viz:
            for (i1, i2), output in two_view_results.items():
                image_i1 = self.scene_optimizer.loader.get_image(i1)
                image_i2 = self.scene_optimizer.loader.get_image(i2)
                viz_utils.save_twoview_correspondences_viz(
                    image_i1,
                    image_i2,
                    keypoints_list[i1],
                    keypoints_list[i2],
                    output.v_corr_idxs,
                    two_view_report=output.post_isp_report,
                    file_path=os.path.join(
                        self.scene_optimizer._plot_correspondence_path,
                        f"{i1}_{i2}__{image_i1.file_name}_{image_i2.file_name}.jpg",
                    ),
                )

    def _aggregate_two_view_metrics(
        self, keypoints_list, two_view_results, correspondence_generation_duration_sec, two_view_estimation_duration_sec
    ):
        self._maybe_save_two_view_viz(keypoints_list, two_view_results)

        post_isp_two_view_reports_dict = {edge: output.post_isp_report for edge, output in two_view_results.items()}
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
        return two_view_agg_metrics

    def _partition_view_graph(self, visibility_graph, two_view_results):
        assert self.graph_partitioner is not None, "Graph partitioner is not set up!"
        subgraphs = self.graph_partitioner.run(visibility_graph)
        if len(subgraphs) == 1:
            # single partition
            self.scene_optimizer.create_output_directories(None)
            return [two_view_results]
        else:
            logger.info("Partitioned into %d subgraphs", len(subgraphs))
            # Group results by subgraph
            return group_results_by_subgraph(two_view_results, subgraphs)

    def _process_subgraph(self, idx, subgraph_two_view_results, keypoints_list, maybe_intrinsics, num_subgraphs):
        logger.info(
            "Creating computation graph for subgraph %d / %d with %d image pairs",
            idx + 1,
            num_subgraphs,
            len(subgraph_two_view_results),
        )
        if num_subgraphs > 1:
            self.scene_optimizer.create_output_directories(idx + 1)

        if len(subgraph_two_view_results) > 0:
            # TODO(Frank): would be nice if relative pose prior was part of TwoViewResult
            # TODO(Frank): I think the loader should compute a Delayed dataclass, or a future

            return self.scene_optimizer.create_computation_graph(
                keypoints_list=keypoints_list,
                two_view_results=subgraph_two_view_results,
                num_images=len(self.scene_optimizer.loader),
                images=self.scene_optimizer.loader.create_computation_graph_for_images(),
                camera_intrinsics=maybe_intrinsics,  # TODO(Frank): really? None is allowed?
                relative_pose_priors=self.scene_optimizer.loader.get_relative_pose_priors(
                    list(subgraph_two_view_results.keys())
                ),
                absolute_pose_priors=self.scene_optimizer.loader.get_absolute_pose_priors(),
                cameras_gt=self.scene_optimizer.loader.get_gt_cameras(),
                gt_wTi_list=self.scene_optimizer.loader.get_gt_poses(),
                gt_scene_mesh=self.scene_optimizer.loader.get_gt_scene_trimesh(),
            )
        else:
            logger.warning(f"Skipping subgraph {idx+1} as it has no valid two-view results.")
            return None, [], []


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


if __name__ == "__main__":
    # Entry point for direct execution
    runner = GtsfmRunner()
    runner.run()
