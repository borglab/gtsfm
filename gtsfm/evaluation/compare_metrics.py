"""Script to log metrics for comparison across SfM pipelines (GTSfM, COLMAP, etc.) as a JSON.

Authors: Jon Womack
"""
import json
import os
from typing import Dict

import gtsfm.utils.io as io_utils
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

import thirdparty.colmap.scripts.python.read_write_model as colmap_io

def compare_metrics(txt_metric_paths: Dict[str, str], json_path: str) -> None:
    """Produces a json file containing data across SfM pipelines formatted as a GTSfMMetricsGroup.

    Args:
        txt_metric_paths: a list of paths to directories containing: cameras.txt, images.txt, and points3D.txt files
    """
    cameras, images, points3d = colmap_io.read_model(path=txt_metric_paths['colmap'], ext=".txt")
    cameras, images, image_files, sfmtracks = io_utils.colmap2gtsfm(cameras, images, points3d, load_sfmtracks=True)

    print(len(images))
    print(len(cameras))
    num_images = len(images)
    num_tracks = len(sfmtracks)

    cameras, images, points3d = colmap_io.read_model(path=txt_metric_paths['gtsfm'], ext=".txt")
    cameras, images, image_files, sfmtracks = io_utils.colmap2gtsfm(cameras, images, points3d, load_sfmtracks=True)

    print(len(images))
    print(len(cameras))
    num_images = len(images)
    num_tracks = len(sfmtracks)




    gtsfm_metrics = []
    for pipeline_name in txt_metric_paths.keys():
        # Add 3D point information
        fpath = os.path.join(txt_metric_paths[pipeline_name], "points3D.txt")
        with open(fpath, "r") as f:
            data = f.readlines()
            points = data[3:]
            track_lengths = []
            for point in points:
                track_lengths.append(len(point.split()[8:]) // 2)

        gtsfm_metrics.append(
            GtsfmMetric(
                "Number of 3D Points " + pipeline_name,
                data[2][data[2].find(":") + 1 : data[2].find(",")],
            )
        )
        gtsfm_metrics.append(
            GtsfmMetric(
                "Mean Track Length " + pipeline_name, data[2][data[2].rindex(":") + 1 :]
            )
        )
        gtsfm_metrics.append(
            GtsfmMetric(
                pipeline_name + "_track_lengths",
                track_lengths,
                store_full_data=False,
                plot_type=GtsfmMetric.PlotType.HISTOGRAM,
            )
        )

        # Add image information
        fpath = os.path.join(txt_metric_paths[pipeline_name], "images.txt")
        with open(fpath, "r") as f:
            data = f.readlines()
            images = data[4:]
        gtsfm_metrics.append(
            GtsfmMetric(
                "Number of Images " + pipeline_name,
                data[3][data[3].find(":") + 1 : data[3].find(",")],
            )
        )
        gtsfm_metrics.append(
            GtsfmMetric(
                "Mean Observations Per Image " + pipeline_name,
                data[3][data[3].rindex(":") + 1 :],
            )
        )

        # Add camera information
        calibrations = io_utils.read_cameras_txt(os.path.join(txt_metric_paths[pipeline_name], "cameras.txt"))
        gtsfm_metrics.append(
            GtsfmMetric(
                "Number of Cameras " + pipeline_name, len(calibrations)
            )
        )

    comparison_metrics = GtsfmMetricsGroup("GTSFM vs. Ground Truth", gtsfm_metrics)
    comparison_metrics.save_to_json(
        os.path.join(json_path, "gt_comparison_metrics.json")
    )
