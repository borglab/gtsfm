"""Base class for runner that executes SfM."""

import argparse
import logging
import os
from pathlib import Path
from typing import cast

import hydra
from dask import config as dask_config
from dask.distributed import Client, LocalCluster, SSHCluster
from hydra.utils import instantiate
from omegaconf import OmegaConf

import gtsfm.utils.logger as logger_utils
from gtsfm.cluster_optimizer import Multiview
from gtsfm.loader.configuration import add_loader_args, build_loader_overrides
from gtsfm.scene_optimizer import SceneOptimizer
from gtsfm.utils.configuration import log_full_configuration

dask_config.set({"distributed.scheduler.worker-ttl": None})

logger = logger_utils.get_logger()

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent

class GtsfmRunner:
    def __init__(self, override_args=None) -> None:
        argparser: argparse.ArgumentParser = self.construct_argparser()
        self.parsed_args, self._hydra_cli_overrides = argparser.parse_known_args(args=override_args)
        if self.parsed_args.dask_tmpdir:
            dask_config.set({"temporary_directory": DEFAULT_OUTPUT_ROOT / self.parsed_args.dask_tmpdir})

        # Configure the logging system
        log_level = getattr(logging, self.parsed_args.log.upper(), None)
        if log_level is not None:
            logger.setLevel(log_level)

        logger.info("🌟 GTSFM: Constructing SceneOptimizer...")
        self.scene_optimizer: SceneOptimizer = self._construct_scene_optimizer()

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="GTSFM Runner")

        parser.add_argument(
            "--config_name",
            type=str,
            default="sift_front_end",
            help="Master config, including back-end configuration. Options include `unified_config.yaml`,"
            " `sift_front_end.yaml`, `deep_front_end.yaml`, etc.",
        )

        # Loader configuration
        add_loader_args(parser)

        # Phase 1 flags: Global Descriptor, Retriever, Graph Partitioner
        parser.add_argument(
            "--global_descriptor_config_name",
            type=str,
            default=None,
            help="Override flag for global descriptor (choose from among gtsfm/configs/global_descriptor).",
        )
        parser.add_argument(
            "--retriever_config_name",
            type=str,
            default=None,
            help="Override flag for retriever (choose from among gtsfm/configs/retriever).",
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
            "--graph_partitioner",
            type=str,
            choices=["single", "binary", "metis"],
            help="Graph partitioner preset to use (see gtsfm/configs/graph_partitioner). "
            "If omitted, each config's default applies.",
        )
        parser.add_argument(
            "--share_intrinsics", action="store_true", help="Shares the intrinsics between all the cameras."
        )

        # Cluster Optimizers
        # MVO flags
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
        parser.add_argument("--run_mvs", action="store_true", help="Run dense MVS reconstruction")
        parser.add_argument("--run_gs", action="store_true", help="Run Gaussian Splatting")
        parser.add_argument(
            "--gaussian_splatting_config_name",
            type=str,
            default="base_gs",
            help="Override flag for your own gaussian splatting implementation.",
        )

        # Logging and output configuration
        parser.add_argument(
            "--output_root",
            type=str,
            default=DEFAULT_OUTPUT_ROOT,
            help="Root directory. Results, plots and metrics will be stored in subdirectories,"
            " e.g. {output_root}/results",
        )
        parser.add_argument(
            "-l",
            "--log",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",  # Set a default level
            help="Set the logging level",
        )

        # SSH Cluster setup
        parser.add_argument(
            "--cluster_config",
            type=str,
            default=None,
            help="config listing IP worker addresses for the cluster,"
            " first worker is used as scheduler and should contain the dataset",
        )
        parser.add_argument(
            "--num_retry_cluster_connection",
            type=int,
            default=3,
            help="Number of times to retry cluster connection if it fails.",
        )

        # Dask configuration
        parser.add_argument(
            "--num_workers", type=int, default=1, help="Number of workers to start (processes, by default)."
        )
        parser.add_argument("--threads_per_worker", type=int, default=1, help="Number of threads per each worker.")
        parser.add_argument(
            "--worker_memory_limit", type=str, default="16GB", help="Memory limit per worker, e.g. `16GB`"
        )
        parser.add_argument("--dashboard_port", type=str, default=":8787", help="dask dashboard port number")
        parser.add_argument(
            "--dask_tmpdir",
            type=str,
            default=None,
            help="tmp directory for dask workers, uses dask's default (/tmp) if not set",
        )

        return parser

    def _construct_scene_optimizer(self) -> SceneOptimizer:
        """Construct scene optimizer.

        All configs are relative to the gtsfm module.
        """
        logger.info(f"📁 Config File: {self.parsed_args.config_name}")
        with hydra.initialize_config_module(config_module="gtsfm.configs", version_base=None):
            overrides = ["+output_root=" + str(self.parsed_args.output_root)]
            if self.parsed_args.share_intrinsics:
                overrides.append("cluster_optimizer.multiview_optimizer.bundle_adjustment_module.shared_calib=True")

            # Loader-related overrides centralized in gtsfm.loader.configuration
            overrides.extend(
                build_loader_overrides(self.parsed_args, default_max_resolution=self.parsed_args.max_resolution)
            )

            if getattr(self.parsed_args, "graph_partitioner", None):
                overrides.append(f"+graph_partitioner={self.parsed_args.graph_partitioner}")

            if getattr(self, "_hydra_cli_overrides", None):
                overrides.extend(self._hydra_cli_overrides)

            main_cfg = hydra.compose(
                config_name=self.parsed_args.config_name,
                overrides=overrides,
            )
        log_full_configuration(main_cfg, logger)
        logger.info("⏳ Instantiating ..")
        scene_optimizer: SceneOptimizer = instantiate(main_cfg)

        # Override retriever.
        if (retriever_config_name := self.parsed_args.retriever_config_name) is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.retriever", version_base=None):
                retriever_cfg = hydra.compose(retriever_config_name)
                logger.info(f"🔄 Applying Retriever Override: {retriever_config_name}")
                scene_optimizer.image_pairs_generator._retriever = instantiate(retriever_cfg.retriever)

        # Override global descriptor.
        if (global_descriptor_config_name := self.parsed_args.global_descriptor_config_name) is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.global_descriptor", version_base=None):
                global_descriptor_cfg = hydra.compose(global_descriptor_config_name)
                logger.info(f"🔄 Applying Global Descriptor Override: {global_descriptor_config_name}")
                scene_optimizer.image_pairs_generator._global_descriptor = instantiate(
                    global_descriptor_cfg.global_descriptor
                )

        # Set retriever specific params if specified with CLI.
        retriever = scene_optimizer.image_pairs_generator._retriever
        if self.parsed_args.max_frame_lookahead is not None:
            try:
                retriever.set_max_frame_lookahead(self.parsed_args.max_frame_lookahead)
                logger.info(f"🔄 Setting max_frame_lookahead: {self.parsed_args.max_frame_lookahead}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to set max_frame_lookahead: {e}")
        if self.parsed_args.num_matched is not None:
            try:
                retriever.set_num_matched(self.parsed_args.num_matched)
                logger.info(f"🔄 Setting num_matched: {self.parsed_args.num_matched}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to set num_matched: {e}")

        # Set flags for the MVO cluster optimizer if applicable.
        cluster_optimizer_is_multiview = isinstance(scene_optimizer.cluster_optimizer, Multiview)

        if cluster_optimizer_is_multiview:
            self._set_mvo_overwrites(scene_optimizer)

        return scene_optimizer

    def _set_mvo_overwrites(self, scene_optimizer: SceneOptimizer) -> None:
        """Set MVO-specific overwrites based on CLI flags."""
        multiview_optimizer = cast(Multiview, scene_optimizer.cluster_optimizer)

        # Override correspondence generator.
        if (correspondence_config_name := self.parsed_args.correspondence_generator_config_name) is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.correspondence", version_base=None):
                correspondence_cfg = hydra.compose(correspondence_config_name)
                logger.info(f"🔄 Applying Correspondence Override: " f"{correspondence_config_name}")
                multiview_optimizer.correspondence_generator = instantiate(correspondence_cfg.CorrespondenceGenerator)

        # Override verifier.
        if (verifier_config_name := self.parsed_args.verifier_config_name) is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.verifier", version_base=None):
                verifier_cfg = hydra.compose(verifier_config_name)
                logger.info(f"🔄 Applying Verifier Override: {verifier_config_name}")
                multiview_optimizer.two_view_estimator._verifier = instantiate(verifier_cfg.verifier)

        # Configure Multiview-specific toggles based on CLI flags.
        if not self.parsed_args.run_mvs:
            multiview_optimizer.dense_multiview_optimizer = None
            logger.info("🔄 Disabled Multiview dense MVS optimizer via CLI flag --run_mvs=False")

        # Override gaussian splatting
        if self.parsed_args.run_gs:
            if (gs_config_name := self.parsed_args.gaussian_splatting_config_name) is not None:
                with hydra.initialize_config_module(
                    config_module="gtsfm.configs.gaussian_splatting", version_base=None
                ):
                    gs_cfg = hydra.compose(gs_config_name)
                    logger.info(f"🔄 Applying Gaussian Splatting Override: " f"{gs_config_name}")
                    multiview_optimizer.gaussian_splatting_optimizer = instantiate(gs_cfg.gaussian_splatting_optimizer)
        else:
            multiview_optimizer.gaussian_splatting_optimizer = None
            logger.info("🔄 Disabled Multiview Gaussian Splatting optimizer via CLI flag --run_gs=False")

    def setup_ssh_cluster_with_retries(self):
        """Sets up SSH Cluster allowing multiple retries upon connection failures."""
        config = OmegaConf.load(os.path.join("gtsfm", "configs", self.parsed_args.cluster_config))
        workers = dict(config)["workers"]
        scheduler = workers[0]
        # Detect GPU cluster configuration
        is_gpu_cluster = any("gpus" in w or "gpu_memory_pool" in w for w in workers)

        connected = False
        retry_count = 0

        while retry_count < self.parsed_args.num_retry_cluster_connection and not connected:
            logger.info(f"Connecting to the cluster: attempt {retry_count + 1}")
            logger.info(f"Using {scheduler} as scheduler")
            logger.info(f"Using {len(workers)} workers")

            if is_gpu_cluster:
                logger.info("🎮 GPU cluster configuration detected")

            try:
                if is_gpu_cluster:
                    cluster = self._create_local_cuda_cluster(workers)
                    logger.info("🎮 GPU cluster created successfully")
                else:
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
                return cluster, config
            except Exception as e:
                logger.info(f"Worker failed to start: {str(e)}")
                retry_count += 1

        if not connected:
            raise ValueError(
                f"Connection to cluster could not be established after {self.parsed_args.num_retry_cluster_connection}"
                " attempts. Aborting..."
            )
        return cluster, config

    def _create_local_cuda_cluster(self, workers):
        """Create LocalCUDACluster for single multi-GPU machine with SLURM.

        SLURM automatically sets CUDA_VISIBLE_DEVICES, so we don't need to
        manually configure GPU IDs. We just need to tell dask-cuda how many
        workers to create and their memory limits.

        Args:
            workers: List of worker configuration dicts with GPU settings

        Returns:
            LocalCUDACluster configured for the specified GPUs
        """
        try:
            from dask_cuda import LocalCUDACluster
        except ImportError:
            raise ImportError("dask-cuda is required for GPU clusters. Install with: pip install dask-cuda")

        n_workers = len(workers)

        # Get configuration from first worker (assuming uniform config)
        use_ucx = workers[0].get("use_ucx", False)
        protocol = "ucx" if use_ucx else "tcp"
        rmm_pool_size = workers[0].get("gpu_memory_pool", None)
        device_memory_limit = workers[0].get("system_memory", "10GB")

        logger.info(f"🎮 Creating LocalCUDACluster with {n_workers} workers")
        logger.info(f"📌 SLURM CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        logger.info(f"📌 Protocol: {protocol} | RMM pool: {rmm_pool_size} | Device memory: {device_memory_limit}")

        cluster_kwargs = {
            "n_workers": n_workers,
            "protocol": protocol,
            "device_memory_limit": device_memory_limit,
            "threads_per_worker": 1,  # One thread per GPU worker
            "dashboard_address": self.parsed_args.dashboard_port,
        }

        # Only add rmm_pool_size if RMM is available and requested
        if rmm_pool_size is not None:
            logger.warning("RMM pool size is not supported in this version of GTSFM.")

        # Add UCX-specific settings if enabled
        if use_ucx:
            cluster_kwargs.update(
                {
                    "enable_tcp_over_ucx": True,
                    "enable_nvlink": True,
                    "enable_infiniband": False,
                }
            )

        cluster = LocalCUDACluster(**cluster_kwargs)
        logger.info("✅ LocalCUDACluster created successfully")

        return cluster

    def _create_dask_client(self):
        if self.parsed_args.cluster_config:
            # Case 2 or 3: Distributed multi-GPU machines
            cluster, config = self.setup_ssh_cluster_with_retries()
            client = Client(cluster)
            client.forward_logging()

            workers = dict(config)["workers"]
            unique_hosts = set(w["host"] for w in workers)

            if len(unique_hosts) > 1:
                io_worker = list(client.scheduler_info()["workers"].keys())[0]
                self.scene_optimizer.loader._input_worker = io_worker
                self.scene_optimizer.cluster_optimizer._output_worker = io_worker
            else:
                logger.info("🖥️  Single-machine multi-GPU cluster")
                logger.info("   All workers can access data locally")
        else:
            # Case 1: Single Local Machine
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

    def run(self) -> None:
        """Just create the client and call scene optimizer."""
        logger.info("🌟 GTSFM: Creating Dask client...")
        client = self._create_dask_client()

        logger.info("🌟 GTSFM: Starting SceneOptimizer...")
        self.scene_optimizer.run(client)

        # Shutdown the Dask client
        logger.info("🌟 GTSFM: Shutting down Dask client...")
        if client is not None:
            client.shutdown()


if __name__ == "__main__":
    # Entry point for direct execution
    runner = GtsfmRunner()
    runner.run()
