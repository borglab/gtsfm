"""Base class for runner that executes SfM."""

import argparse
import os
import time
from abc import abstractmethod, abstractproperty
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dask
import hydra
import numpy as np
from dask import config as dask_config
from dask.distributed import Client, LocalCluster, SSHCluster, performance_report
from gtsam import Rot3, Unit3
from hydra.utils import instantiate
from omegaconf import OmegaConf

import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.viz as viz_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm import two_view_estimator
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import ImageMatchingRegime
from gtsfm.scene_optimizer import SceneOptimizer
from gtsfm.two_view_estimator import TWO_VIEW_OUTPUT, TwoViewEstimationReport, run_two_view_estimator_as_futures
from gtsfm.ui.process_graph_generator import ProcessGraphGenerator

dask_config.set({"distributed.scheduler.worker-ttl": None})

logger = logger_utils.get_logger()

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent
REACT_METRICS_PATH = DEFAULT_OUTPUT_ROOT / "rtf_vis_tool" / "src" / "result_metrics"


class GtsfmRunnerBase:
    @abstractproperty
    def tag(self):
        pass

    def __init__(self, override_args: Any = None) -> None:
        argparser: argparse.ArgumentParser = self.construct_argparser()
        self.parsed_args: argparse.Namespace = argparser.parse_args(args=override_args)
        if self.parsed_args.dask_tmpdir:
            dask.config.set({"temporary_directory": DEFAULT_OUTPUT_ROOT / self.parsed_args.dask_tmpdir})

        self.loader: LoaderBase = self.construct_loader()
        self.scene_optimizer: SceneOptimizer = self.construct_scene_optimizer()

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
        parser.add_argument("--mvs_off", action="store_true", help="Turn off dense MVS reconstruction")
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
                scene_optimizer.retriever = instantiate(retriever_cfg.retriever)

        if self.parsed_args.mvs_off:
            scene_optimizer.run_dense_optimizer = False

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
            }
            if self.parsed_args.worker_memory_limit is not None:
                local_cluster_kwargs["memory_limit"] = self.parsed_args.worker_memory_limit
            cluster = LocalCluster(**local_cluster_kwargs)
            client = Client(cluster)

        # Create process graph.
        process_graph_generator = ProcessGraphGenerator()
        if isinstance(self.scene_optimizer.correspondence_generator, ImageCorrespondenceGenerator):
            process_graph_generator.is_image_correspondence = True
        process_graph_generator.save_graph()

        COMBINATIONS = [
            (0,0), (0,5), (0,10),
            (5,0), (5,5), (5,10),
            (10,0), (10,5), (10,10),
        ]

        for (num_matched, max_frame_lookahead) in COMBINATIONS:

            if max_frame_lookahead is not None:
                if scene_optimizer.retriever._matching_regime in [
                    ImageMatchingRegime.SEQUENTIAL,
                    ImageMatchingRegime.SEQUENTIAL_HILTI,
                ]:
                    scene_optimizer.retriever._max_frame_lookahead = max_frame_lookahead
                elif scene_optimizer.retriever._matching_regime == ImageMatchingRegime.SEQUENTIAL_WITH_RETRIEVAL:
                    scene_optimizer.retriever._seq_retriever._max_frame_lookahead = max_frame_lookahead
                else:
                    raise ValueError(
                        "`max_frame_lookahead` arg is incompatible with retriever matching regime "
                        f"{scene_optimizer.retriever._matching_regime}"
                    )
            if num_matched is not None:
                if scene_optimizer.retriever._matching_regime == ImageMatchingRegime.SEQUENTIAL_WITH_RETRIEVAL:
                    scene_optimizer.retriever._similarity_retriever._num_matched = num_matched
                elif scene_optimizer.retriever._matching_regime == ImageMatchingRegime.RETRIEVAL:
                    scene_optimizer.retriever._num_matched = num_matched
                else:
                    raise ValueError(
                        "`num_matched` arg is incompatible with retriever matching regime "
                        f"{scene_optimizer.retriever._matching_regime}"
                    )

            self.run_with_retriever_setting()

        # Loop over all runs, see which has highest median track length.

        experiment_roots = []
        for experiment_root in experiment_roots:

            dirpath = Path(experiment_root) / "result_metrics"
            frontend_name = Path(experiment_root).name
            table["method_name"].append(frontend_name)

            json_fname = ""
            metric_name = ""

            for json_fname, metric_names, nickname in zip(
                SECTION_FILE_NAMES,
                SECTION_METRIC_LISTS,
                SECTION_NICKNAMES,
            ):
                section_name = Path(json_fname).stem
                print(f"{dirpath}/{json_fname}")
                json_data = io_utils.read_json_file(f"{dirpath}/{json_fname}")[section_name]
                for metric_name in metric_names:
                    full_metric_name = f"{nickname}_" + " ".join(metric_name.split("_"))
                    if method_idx == 0:
                        headers.append(full_metric_name)

                    if "pose_auc_" in metric_name and metric_name in SCALAR_METRIC_NAMES:
                        table[full_metric_name].append(json_data[metric_name] * 100)
                    elif metric_name in SCALAR_METRIC_NAMES:
                        print(f"{metric_name}: {json_data[metric_name]}")
                        table[full_metric_name].append(json_data[metric_name])
                    else:
                        med = f"{json_data[metric_name]['summary']['median']:.2f}"




    def run_with_retriever_setting(self) -> GtsfmData:
        start_time = time.time()

        # TODO(Ayush): Use futures
        retriever_start_time = time.time()
        pairs_graph = self.scene_optimizer.retriever.create_computation_graph(
            self.loader, plots_output_dir=self.scene_optimizer._plot_base_path
        )
        with performance_report(filename="retriever-dask-report.html"):
            image_pair_indices = pairs_graph.compute()

        retriever_metrics = self.scene_optimizer.retriever.evaluate(self.loader, image_pair_indices)
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

        i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, two_view_reports_dict = unzip_two_view_results(two_view_results_dict)

        if self.scene_optimizer._save_two_view_correspondences_viz:
            for (i1, i2) in v_corr_idxs_dict.keys():
                image_i1 = self.loader.get_image(i1)
                image_i2 = self.loader.get_image(i2)
                viz_utils.save_twoview_correspondences_viz(
                    image_i1,
                    image_i2,
                    keypoints_list[i1],
                    keypoints_list[i2],
                    v_corr_idxs_dict[(i1, i2)],
                    two_view_report=two_view_reports_dict[(i1, i2)],
                    file_path=os.path.join(
                        self.scene_optimizer._plot_correspondence_path,
                        f"{i1}_{i2}__{image_i1.file_name}_{image_i2.file_name}.jpg",
                    ),
                )

        two_view_agg_metrics = two_view_estimator.aggregate_frontend_metrics(
            two_view_reports_dict=two_view_reports_dict,
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

        delayed_sfm_result, delayed_io, delayed_mvo_metrics_groups = self.scene_optimizer.create_computation_graph(
            keypoints_list=keypoints_list,
            i2Ri1_dict=i2Ri1_dict,
            i2Ui1_dict=i2Ui1_dict,
            v_corr_idxs_dict=v_corr_idxs_dict,
            two_view_reports=two_view_reports_dict,
            num_images=len(self.loader),
            images=self.loader.create_computation_graph_for_images(),
            camera_intrinsics=intrinsics,
            relative_pose_priors=self.loader.get_relative_pose_priors(image_pair_indices),
            absolute_pose_priors=self.loader.get_absolute_pose_priors(),
            cameras_gt=self.loader.get_gt_cameras(),
            gt_wTi_list=self.loader.get_gt_poses(),
            gt_scene_mesh=self.loader.get_gt_scene_trimesh(),
        )

        with performance_report(filename="scene-optimizer-dask-report.html"):
            sfm_result, *other_results = dask.compute(delayed_sfm_result, *delayed_io, *delayed_mvo_metrics_groups)
        mvo_metrics_groups = [x for x in other_results if isinstance(x, GtsfmMetricsGroup)]

        assert isinstance(sfm_result, GtsfmData)
        all_metrics_groups.extend(mvo_metrics_groups)

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)

        total_summary_metrics = GtsfmMetricsGroup(
            "total_summary_metrics", [GtsfmMetric("total_runtime_sec", duration_sec)]
        )
        all_metrics_groups.append(total_summary_metrics)

        save_metrics_reports(all_metrics_groups, os.path.join(self.scene_optimizer.output_root, "result_metrics"))
        return sfm_result


def unzip_two_view_results(
    two_view_results: Dict[Tuple[int, int], TWO_VIEW_OUTPUT]
) -> Tuple[
    Dict[Tuple[int, int], Rot3],
    Dict[Tuple[int, int], Unit3],
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], TwoViewEstimationReport],
]:
    """Unzip the tuple TWO_VIEW_OUTPUT into 1 dictionary for 1 element in the tuple."""
    i2Ri1_dict: Dict[Tuple[int, int], Rot3] = {}
    i2Ui1_dict: Dict[Tuple[int, int], Unit3] = {}
    v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray] = {}
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport] = {}

    for (i1, i2), two_view_output in two_view_results.items():
        i2Ri1 = two_view_output[0]
        i2Ui1 = two_view_output[1]
        if i2Ri1 is None or i2Ui1 is None:
            continue

        i2Ri1_dict[(i1, i2)] = i2Ri1
        i2Ui1_dict[(i1, i2)] = i2Ui1
        v_corr_idxs_dict[(i1, i2)] = two_view_output[2]
        two_view_reports_dict[(i1, i2)] = two_view_output[5]

    return i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, two_view_reports_dict


def save_metrics_reports(metrics_group_list: List[GtsfmMetricsGroup], metrics_path: str) -> None:
    """Saves metrics to JSON and HTML report.

    Args:
        metrics_graph: List of GtsfmMetricsGroup from different modules wrapped as Delayed.
        metrics_path: Path to directory where computed metrics will be saved.
    """

    # Save metrics to JSON
    metrics_utils.save_metrics_as_json(metrics_group_list, metrics_path)
    metrics_utils.save_metrics_as_json(metrics_group_list, str(REACT_METRICS_PATH))

    metrics_report.generate_metrics_report_html(
        metrics_group_list, os.path.join(metrics_path, "gtsfm_metrics_report.html"), None
    )
