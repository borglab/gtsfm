"""Function supporting metric comparisons across SfM pipelines (GTSfM, COLMAP, etc.)

This function converts outputs from SfM pipelines into a format matching
metrics from GTSfM.

Authors: Jon Womack
"""
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
    """Converts the outputs of other SfM pipelines to GTSfMMetricsGroups saved as json files.

    Creates folders for each additional SfM pipeline that contain GTSfMMetricsGroups
    containing the same metrics as GTSFM_MODULE_METRICS_FNAMES. If one of the GTSfM metrics
    is not available from another SfM pipeline, then the metric is left blank for that pipeline.

    Args:
        txt_metric_paths: a list of paths to directories containing outputs of other SfM pipelines
          in COLMAP format i.e. cameras.txt, images.txt, and points3D.txt files.
        json_path: Path to folder that contains metrics as json files.
        GTSFM_MODULE_METRICS_FNAMES: List of GTSfM metrics filenames.
    """
    for pipeline_name in txt_metric_paths.keys():
        cameras, images, points3d = colmap_io.read_model(path=txt_metric_paths[pipeline_name], ext=".txt")
        cameras, images, image_files, sfmtracks = io_utils.colmap2gtsfm(cameras, images, points3d, load_sfmtracks=True)
        num_cameras = len(cameras)
        track_lengths = []
        image_id_num_measurements = {}
        for track in sfmtracks:
            track_lengths.append(track.number_measurements())
            for k in range(track.number_measurements()):
                image_id, uv_measured = track.measurement(k)
                if image_id not in image_id_num_measurements:
                    image_id_num_measurements[image_id] = 1
                else:
                    image_id_num_measurements[image_id] += 1

        colmap2gtsfm = {
            "number_cameras": GtsfmMetric("number_cameras", num_cameras),
            "3d_tracks_length": GtsfmMetric(
                "3d_tracks_length",
                track_lengths,
                plot_type=GtsfmMetric.PlotType.HISTOGRAM,
            ),
        }

        # Create comparable result_metric json for COLMAP
        for filename in GTSFM_MODULE_METRICS_FNAMES:
            metrics = []
            metrics_group = GtsfmMetricsGroup.parse_from_json(os.path.join(json_path, filename))
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
            new_metrics_group.save_to_json(os.path.join(json_path, pipeline_name, os.path.basename(filename)))
