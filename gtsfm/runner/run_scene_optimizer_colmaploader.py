import argparse
import time

import hydra
from dask.distributed import Client, LocalCluster, performance_report
from hydra.utils import instantiate

import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.colmap_loader import ColmapLoader
from gtsfm.scene_optimizer import SceneOptimizer

logger = logger_utils.get_logger()


# fmt: off
# Successive relaxation threshold pairs -- from strictest to loosest
# `inlier ratio` is the minimum allowed inlier ratio w.r.t. the estimated model
NUM_INLIERS_THRESHOLDS      =  [200, 175, 150, 125, 100, 75,  50,   25, 15] # noqa
MIN_INLIER_RATIOS_THRESHOLDS = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1] # noqa
# fmt: on


def run_scene_optimizer(args: argparse.Namespace) -> None:
    """We solve the problem at varying level of difficulties, starting at the strictest
    setting, and gradually relaxing the problem until a sufficient number of inliers can be found.
    As for measurements that are fed to the backend, we require three times the number of input
    images, for sufficient redundancy in the graph.
    """
    # create dask client only once, and will be re-used for all relaxations
    cluster = LocalCluster(n_workers=args.num_workers, threads_per_worker=args.threads_per_worker)

    start = time.time()
    with hydra.initialize_config_module(config_module="gtsfm.configs"):
        # config is relative to the gtsfm module
        cfg = hydra.compose(config_name=args.config_name)

        scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)
        loader = ColmapLoader(
            colmap_files_dirpath=args.colmap_files_dirpath,
            images_dir=args.images_dir,
            max_frame_lookahead=args.max_frame_lookahead,
        )

        # try to relax the problem repeatedly
        for (num_inliers_required, min_allowed_inlier_ratio_est_model) in zip(
            NUM_INLIERS_THRESHOLDS, MIN_INLIER_RATIOS_THRESHOLDS
        ):
            scene_optimizer.two_view_estimator.min_num_inliers_acceptance = num_inliers_required
            scene_optimizer.two_view_estimator._verifier.min_allowed_inlier_ratio_est_model = (
                min_allowed_inlier_ratio_est_model
            )

            logger.info("New threshold:  %d inliers", scene_optimizer.two_view_estimator.min_num_inliers_acceptance)
            logger.info(
                "New threshold: %.f inlier ratio",
                scene_optimizer.two_view_estimator._verifier.min_allowed_inlier_ratio_est_model,
            )

            sfm_result_graph = scene_optimizer.create_computation_graph(
                num_images=len(loader),
                image_pair_indices=loader.get_valid_pairs(),
                image_graph=loader.create_computation_graph_for_images(),
                camera_intrinsics_graph=loader.create_computation_graph_for_intrinsics(),
                image_shape_graph=loader.create_computation_graph_for_image_shapes(),
                gt_pose_graph=loader.create_computation_graph_for_poses(),
            )

            try:
                with Client(cluster), performance_report(filename="dask-report.html"):
                    sfm_result = sfm_result_graph.compute()
                assert isinstance(sfm_result, GtsfmData)

                # check for success
                frontend_result = io_utils.read_json_file("result_metrics/cycle_consistent_frontend_summary.json")
                num_backend_input_pairs = frontend_result["cycle_consistent_frontend_summary"]["num_valid_image_pairs"]
                num_required_backend_input_pairs = 3 * len(loader)
                if num_backend_input_pairs < num_required_backend_input_pairs:
                    logger.info("Too few measurements at this threshold, will try relaxing the problem...")
                    logger.info(
                        "Found only %d num_backend_input_pairs, needed %d",
                        num_backend_input_pairs,
                        num_required_backend_input_pairs,
                    )
                else:
                    break

            except Exception as e:
                logger.exception("Failed")
                print("Computation failed, will try relaxing the problem ...")

    end = time.time()
    duration_sec = end - start
    logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GTSFM with intrinsics and image names stored in COLMAP-format")
    parser.add_argument(
        "--images_dir", type=str, required=True, help="path to directory containing png, jpeg, or jpg images files"
    )
    parser.add_argument(
        "--colmap_files_dirpath",
        type=str,
        required=True,
        help="path to directory containing images.txt, points3D.txt, and cameras.txt",
    )
    parser.add_argument(
        "--max_frame_lookahead",
        type=int,
        default=1,
        help="maximum number of consecutive frames to consider for matching/co-visibility",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to start (processes, by default)",
    )
    parser.add_argument(
        "--threads_per_worker",
        type=int,
        default=1,
        help="Number of threads per each worker",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="deep_front_end.yaml",
        help="Choose sift_front_end.yaml or deep_front_end.yaml",
    )

    args = parser.parse_args()
    run_scene_optimizer(args)
