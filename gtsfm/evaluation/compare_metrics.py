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


def compare_metrics(
    txt_metric_paths: Dict[str, str],
    json_path: str,
    GTSFM_MODULE_METRICS_FNAMES: List[str],
) -> None:
    """Produces json files containing data from other SfM pipelines formatted as common GTSfMMetricsGroups.

    Args:
        txt_metric_paths: a list of paths to directories containing: cameras.txt, images.txt, and points3D.txt files
    """
    for pipeline_name in txt_metric_paths.keys():
        cameras, images, points3d = colmap_io.read_model(
            path=txt_metric_paths[pipeline_name], ext=".txt"
        )
        cameras, images, image_files, sfmtracks = io_utils.colmap2gtsfm(
            cameras, images, points3d, load_sfmtracks=True
        )
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
        num_total_frontend_measurements = sum(image_id_num_measurements.values())

        colmap2gtsfm = {
            "number_cameras": GtsfmMetric("number_cameras", num_cameras),
            "3d_tracks_length": GtsfmMetric(
                "3d_tracks_length",
                track_lengths,
                plot_type=GtsfmMetric.PlotType.HISTOGRAM,
            )
            #     ),
            # "num_total_frontend_measurements":
            #     GtsfmMetric("num_total_frontend_measurements", num_total_frontend_measurements)
        }

        # Create comparable result_metric json for COLMAP
        for filename in GTSFM_MODULE_METRICS_FNAMES:
            metrics = []
            metrics_group = GtsfmMetricsGroup.parse_from_json(
                os.path.join(json_path, filename)
            )
            for metric in metrics_group.metrics:
                # Case 1: mapping from COLMAP to GTSfM is known
                if metric.name in colmap2gtsfm.keys():
                    metrics.append(colmap2gtsfm[metric.name])
                # Case 2: mapping from COLMAP to GTSfM is known
                else:
                    if metric._dim == 1:
                        # Case 2a: dict summary
                        metrics.append(GtsfmMetric(metric.name, []))
                    else:
                        # Case 2b: scalar metric
                        metrics.append(GtsfmMetric(metric.name, ""))
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
    #             "Mean Observations Per Image " + pipeline_name, np.mean(observations_per_image)
    #         )
    #     )
    #     gtsfm_metrics.append(
    #         GtsfmMetric(
    #             "Median Observations Per Image " + pipeline_name, np.median(observations_per_image)
    #         )
    #     )
