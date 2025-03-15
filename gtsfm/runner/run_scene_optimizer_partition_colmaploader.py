#!/usr/bin/env python3
"""
Run GTSFM with the ColmapLoader and scene partitioning.

Authors: Zongyue Liu
"""

import argparse
import os
import time
from pathlib import Path

import dask
import hydra
from dask.distributed import Client, LocalCluster, performance_report
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np
import torch
from typing import List, Tuple

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.bundle.global_ba import GlobalBundleAdjustment
from gtsfm.data_association.data_assoc import DataAssociation
from gtsfm.evaluation.metrics import GtsfmMetricsGroup, GtsfmMetric
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.frontend.matcher.superglue_matcher import SuperGlueMatcher
from gtsfm.loader.colmap_loader import ColmapLoader
from gtsfm.retriever.similarity_image_pairs_generator import SimilarityImagePairsGenerator
from gtsfm.retriever.image_pairs_generator import ImagePairsGenerator
from gtsfm.retriever.netvlad_retriever import NetVLADRetriever
from gtsfm.retriever.sequential_retriever import SequentialRetriever
from gtsfm.retriever.joint_netvlad_sequential_retriever import JointNetVLADSequentialRetriever
from gtsfm.frontend.global_descriptor.netvlad_global_descriptor import NetVLADGlobalDescriptor
from gtsfm.scene_optimizer import SceneOptimizer
from gtsfm.two_view_estimator import TwoViewEstimator
from gtsfm.multi_view_optimizer import MultiViewOptimizer
from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.graph_partitioner.single_partition import SinglePartition
from gtsfm.runner.scene_optimizer_helper import run_scene_optimizer_for_subgraphs
from gtsfm.runner.gtsfm_runner_base import run_two_view_estimator_as_futures, unzip_two_view_results, save_metrics_reports
import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GTSFM with partitioning using ColmapLoader")
    
    parser.add_argument(
        "--colmap_path",
        type=str,
        required=True,
        help="Path to Colmap output directory containing cameras.txt, images.txt, and points3D.txt",
        dest="colmap_files_dirpath",
    )
    
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to directory containing images",
        dest="images_dir",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory path",
    )
    
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=1000,
        help="Maximum image resolution (longest side)",
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of Dask workers",
    )
    
    parser.add_argument(
        "--threads_per_worker",
        type=int,
        default=1,
        help="Number of threads per Dask worker",
    )
    
    parser.add_argument(
        "--dashboard_port",
        type=str,
        default=":8787",
        help="Dashboard port for Dask",
    )
    
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.0,
        help="Threshold for similarity matrix",
    )
    
    parser.add_argument(
        "--num_matched",
        type=int,
        default=5,
        help="Number of matched pairs per image",
    )
    
    parser.add_argument(
        "--retriever_type",
        type=str,
        default="netvlad",
        choices=["netvlad", "sequential", "joint"],
        help="Type of retriever to use",
    )
    
    parser.add_argument(
        "--config_name",
        type=str,
        default="sift_front_end",
        help="Name of the Hydra config file to use",
    )
    
    parser.add_argument(
        "--partition_size",
        type=int,
        default=None,
        help="Maximum number of images in a partition (None for single partition)",
    )
    
    return parser.parse_args()


def main():
    """Main function to run GTSFM with partitioning."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize loader
    loader = ColmapLoader(
        colmap_files_dirpath=args.colmap_files_dirpath,
        images_dir=args.images_dir,
        max_resolution=args.max_resolution,
    )
    
    # Initialize retriever based on type
    if args.retriever_type == "netvlad":
        retriever = NetVLADRetriever(num_matched=args.num_matched)
        global_descriptor = NetVLADGlobalDescriptor()

        image_pairs_generator = SimilarityImagePairsGenerator(
            retriever=retriever,
            global_descriptor=global_descriptor
        )
    elif args.retriever_type == "sequential":
        retriever = SequentialRetriever(window_size=args.num_matched)

        image_pairs_generator = SimilarityImagePairsGenerator(
            retriever=retriever
        )
    elif args.retriever_type == "joint":
        retriever = JointNetVLADSequentialRetriever(
            num_matched=args.num_matched,
            window_size=args.num_matched
        )
        global_descriptor = NetVLADGlobalDescriptor()

        image_pairs_generator = SimilarityImagePairsGenerator(
            retriever=retriever,
            global_descriptor=global_descriptor
        )
    else:
        raise ValueError(f"Unknown retriever type: {args.retriever_type}")
    
    # Initialize scene optimizer using Hydra configuration
    scene_optimizer = None
    with hydra.initialize_config_module(config_module="gtsfm.configs", version_base=None):
        try:
            logger.info(f"Trying to load config '{args.config_name}'")
            overrides = [f"+SceneOptimizer.output_root={str(output_dir)}"]
            
            main_cfg = hydra.compose(
                config_name=args.config_name,
                overrides=overrides,
            )
            scene_optimizer = instantiate(main_cfg.SceneOptimizer)
            # Replace the image_pairs_generator with our custom one
            scene_optimizer.image_pairs_generator = image_pairs_generator
            scene_optimizer._plot_base_path = str(output_dir)
            
            logger.info(f"Successfully loaded config '{args.config_name}'")
        except Exception as e:
            logger.warning(f"Failed to load config '{args.config_name}': {e}")

            logger.info("Trying fallback to 'sift_front_end'")
            main_cfg = hydra.compose(
                config_name="sift_front_end",
                overrides=overrides,
            )
            scene_optimizer = instantiate(main_cfg.SceneOptimizer)
            # Replace the image_pairs_generator with our custom one
            scene_optimizer.image_pairs_generator = image_pairs_generator
            scene_optimizer._plot_base_path = str(output_dir)
            
            logger.info("Successfully loaded 'sift_front_end' config")
  
    
    # Initialize graph partitioner
    if args.partition_size is None:
        graph_partitioner = SinglePartition()

    
    # Start time measurement
    start_time = time.time()
    
    # Create Dask cluster
    local_cluster_kwargs = {
        "n_workers": args.num_workers,
        "threads_per_worker": args.threads_per_worker,
        "dashboard_address": args.dashboard_port,
    }
    cluster = LocalCluster(**local_cluster_kwargs)
    client = Client(cluster)
    
    try:
        # Load images as Dask futures
        image_futures = loader.get_all_images_as_futures(client)
        
        # All metrics groups
        all_metrics_groups = []
        
        # Generate image pairs and similarity matrix
        retriever_start_time = time.time()
        with performance_report(filename=str(output_dir / "retriever-dask-report.html")):
            # Convert string path to Path object to avoid path joining issues
            plot_path = Path(scene_optimizer._plot_base_path) if isinstance(scene_optimizer._plot_base_path, str) else scene_optimizer._plot_base_path

            image_pair_indices, similarity_matrix = image_pairs_generator.generate_image_pairs_with_similarity(
                client=client,
                images=image_futures,
                image_fnames=loader.image_filenames(),
                plots_output_dir=plot_path
            )
            
 
            if isinstance(graph_partitioner, SinglePartition) and args.partition_size is None:
                logger.info("Using SinglePartition: no need to modify similarity matrix")

            else:
                logger.info(f"Using partitioner: {type(graph_partitioner).__name__}")

        
        retriever_duration_sec = time.time() - retriever_start_time
        retriever_metrics = image_pairs_generator._retriever.evaluate(len(loader), image_pair_indices)
        retriever_metrics.add_metric(GtsfmMetric("retriever_duration_sec", retriever_duration_sec))
        logger.info(f"Image pair retrieval took {retriever_duration_sec:.2f} sec.")
        all_metrics_groups.append(retriever_metrics)
        
        # Get camera intrinsics
        intrinsics = loader.get_all_intrinsics()
        
        # Generate correspondences and estimate two-view geometry
        with performance_report(filename=str(output_dir / "correspondence-generator-dask-report.html")):
            correspondence_generation_start_time = time.time()
            keypoints_list, putative_corr_idxs_dict = scene_optimizer.correspondence_generator.generate_correspondences(
                client,
                image_futures,
                image_pair_indices,
            )
            correspondence_generation_duration_sec = time.time() - correspondence_generation_start_time
            
            two_view_estimation_start_time = time.time()
            two_view_results_dict = run_two_view_estimator_as_futures(
                client,
                scene_optimizer.two_view_estimator,
                keypoints_list,
                putative_corr_idxs_dict,
                intrinsics,
                loader.get_relative_pose_priors(image_pair_indices),
                loader.get_gt_cameras(),
                gt_scene_mesh=loader.get_gt_scene_trimesh(),
            )
            two_view_estimation_duration_sec = time.time() - two_view_estimation_start_time
        
        i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, _, two_view_reports_dict = unzip_two_view_results(
            two_view_results_dict
        )
        
        # Run scene optimizer for each subgraph
        with performance_report(filename=str(output_dir / "scene-optimizer-dask-report.html")):
            subgraph_results = run_scene_optimizer_for_subgraphs(
                client=client,
                scene_optimizer=scene_optimizer,
                graph_partitioner=graph_partitioner,
                similarity_matrix=similarity_matrix,
                keypoints_list=keypoints_list,
                i2Ri1_dict=i2Ri1_dict,
                i2Ui1_dict=i2Ui1_dict,
                v_corr_idxs_dict=v_corr_idxs_dict,
                two_view_reports=two_view_reports_dict,
                num_images=len(loader),
                images=loader.create_computation_graph_for_images(),
                camera_intrinsics=intrinsics,
                absolute_pose_priors=loader.get_absolute_pose_priors(),
                relative_pose_priors=loader.get_relative_pose_priors(image_pair_indices),
                cameras_gt=loader.get_gt_cameras(),
                gt_wTi_list=loader.get_gt_poses(),
                gt_scene_mesh=loader.get_gt_scene_trimesh(),
            )
        
        
        # Calculate and log run time
        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info(f"GTSFM took {duration_sec / 60:.2f} minutes to compute with partitioning.")
        
        # Save metrics
        total_summary_metrics = GtsfmMetricsGroup(
            "total_summary_metrics", 
            [GtsfmMetric("total_runtime_sec", duration_sec)]
        )
        all_metrics_groups.append(total_summary_metrics)
        
        # Save metrics to disk
        save_metrics_reports(all_metrics_groups, str(output_dir / "result_metrics"))
        
        return None

    finally:
        # Clean up Dask client
        client.shutdown()


if __name__ == "__main__":
    main()