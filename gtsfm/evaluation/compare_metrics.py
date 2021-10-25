"""Script to log metrics for comparison across SfM pipelines (GTSfM, COLMAP, etc.) as a JSON.

Authors: Jon Womack
"""
import json
import numpy as np
import os
from typing import Dict, List

import gtsfm.utils.io as io_utils
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

import thirdparty.colmap.scripts.python.read_write_model as colmap_io

def compare_metrics(txt_metric_paths: Dict[str, str], json_path: str, GTSFM_MODULE_METRICS_FNAMES: List[str]) -> None:
    """Produces json files containing data from other SfM pipelines formatted as common GTSfMMetricsGroups.

    Args:
        txt_metric_paths: a list of paths to directories containing: cameras.txt, images.txt, and points3D.txt files
    """
    for pipeline_name in txt_metric_paths.keys():
        cameras, images, points3d = colmap_io.read_model(path=txt_metric_paths[pipeline_name], ext=".txt")
        cameras, images, image_files, sfmtracks = io_utils.colmap2gtsfm(cameras, images, points3d, load_sfmtracks=True)
        num_images = len(images)
        num_cameras = len(cameras)
        num_tracks = len(sfmtracks)
        # mean_observation_per_image =
        track_lengths = []
        image_id_num_measurements = {}
        for track in sfmtracks:
            track_lengths.append(track.number_measurements())
            for k in range(track.number_measurements()):
                image_id, uv_measured = track.measurement(k)
                if not image_id in image_id_num_measurements:
                    image_id_num_measurements[image_id] = 1
                else:
                    image_id_num_measurements[image_id] += 1
        observations_per_image = list(image_id_num_measurements.values())

        #Create comparable result_metric json for COLMAP
        for filename in GTSFM_MODULE_METRICS_FNAMES:
            metrics = []
            metrics_group = GtsfmMetricsGroup.parse_from_json(os.path.join(json_path, filename))
            for metric in metrics_group.metrics:
                metrics.append(GtsfmMetric(
                    metric.name, "TODO"
                ))
            new_metrics_group = GtsfmMetricsGroup(metrics_group.name, metrics)

            os.makedirs(os.path.join(json_path, pipeline_name), exist_ok=True)
            new_metrics_group.save_to_json(
                os.path.join(json_path, pipeline_name, os.path.basename(filename))
            )


    #
    #     gtsfm_metrics.append(
    #         GtsfmMetric(
    #             "Number of 3D Points " + pipeline_name, num_tracks
    #         )
    #     )
    #     gtsfm_metrics.append(
    #         GtsfmMetric(
    #             "Number of Images " + pipeline_name, num_images
    #         )
    #     )
    #     gtsfm_metrics.append(
    #         GtsfmMetric(
    #             "Number of Cameras " + pipeline_name,  num_cameras
    #         )
    #     )
    #     gtsfm_metrics.append(
    #         GtsfmMetric(
    #             pipeline_name + "_track_lengths",
    #             track_lengths,
    #             store_full_data=False,
    #             plot_type=GtsfmMetric.PlotType.HISTOGRAM,
    #         )
    #     )
    #     gtsfm_metrics.append(
    #         GtsfmMetric(
    #             "Mean Observations Per Image " + pipeline_name, np.mean(observations_per_image)
    #         )
    #     )
    #     gtsfm_metrics.append(
    #         GtsfmMetric(
    #             "Median Observations Per Image " + pipeline_name, np.median(observations_per_image)
    #         )
    #     )
    #
    # comparison_metrics = GtsfmMetricsGroup("GTSFM vs. Ground Truth", gtsfm_metrics)
    # comparison_metrics.save_to_json(
    #     os.path.join(json_path, "gt_comparison_metrics.json")
    # )
