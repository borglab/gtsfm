"""Function supporting metric comparisons across SfM pipelines (GTSfM, COLMAP, etc.)

This function converts outputs from SfM pipelines into a format matching
metrics from GTSfM.

Authors: Jon Womack
"""
import os
from typing import Dict, List

import numpy as np

import gtsfm.utils.io as io_utils
import thirdparty.colmap.scripts.python.read_write_model as colmap_io
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup


def compute_metrics_from_txt(
    cameras: Dict[colmap_io.Camera, int],
    images: Dict[colmap_io.Image, int],
    points3d: Dict[colmap_io.Point3D, int],
    reproj_error_threshold: int,
):
    """Calculate metrics from pipeline outputs parsed from COLMAP txt format.

    Args:
        cameras: Dictionary of COLMAP-formatted Cameras.
        images: Dictionary of COLMAP-formatted Images.
        points3D: Dictionary of COLMAP-formatted Point3Ds.
        reproj_error_threshold: Reprojection error threshold for filtering tracks.

    Returns:
        other_pipeline_metrics: A dictionary of metrics from another pipeline that are comparable with GTSfM
    """
    _, _, intrinsics_gtsfm, sfmtracks, _, _, _ = io_utils.colmap2gtsfm(cameras, images, points3d, load_sfmtracks=True)

    num_cameras = len(intrinsics_gtsfm)
    unfiltered_track_lengths = []
    image_id_num_measurements = {}
    for track in sfmtracks:
        unfiltered_track_lengths.append(track.numberMeasurements())
        for k in range(track.numberMeasurements()):
            image_id, uv_measured = track.measurement(k)
            if image_id not in image_id_num_measurements:
                image_id_num_measurements[image_id] = 1
            else:
                image_id_num_measurements[image_id] += 1

    # Note: IDs begin at 1, so id-1 for indexing list
    unfiltered_reproj_errors = []
    filtered_reproj_errors = []
    filtered_track_lengths = []
    for point3d_id, point3d in points3d.items():
        reproj_error = point3d.error
        unfiltered_reproj_errors.append(reproj_error)
        if reproj_error < reproj_error_threshold:
            filtered_reproj_errors.append(reproj_error)
            filtered_track_lengths.append(len(point3d.image_ids))
    num_filtered_tracks = len(filtered_track_lengths)
    other_pipeline_metrics = {
        "number_cameras": GtsfmMetric("number_cameras", num_cameras),
        "3d_track_lengths_unfiltered": GtsfmMetric(
            "3d_track_lengths_unfiltered",
            np.asarray(
                unfiltered_track_lengths,
            ),
            plot_type=GtsfmMetric.PlotType.HISTOGRAM,
        ),
        "number_tracks_unfiltered": GtsfmMetric("number_tracks_unfiltered", len(sfmtracks)),
        "reprojection_errors_unfiltered_px": GtsfmMetric(
            "reprojection_errors_unfiltered_px",
            unfiltered_reproj_errors,
            plot_type=GtsfmMetric.PlotType.BOX,
        ),
        "3d_track_lengths_filtered": GtsfmMetric(
            "3d_track_lengths_filtered",
            np.asarray(filtered_track_lengths),
            plot_type=GtsfmMetric.PlotType.HISTOGRAM,
        ),
        "number_tracks_filtered": GtsfmMetric("number_tracks_filtered", num_filtered_tracks),
        "reprojection_errors_filtered_px": GtsfmMetric(
            "reprojection_errors_filtered_px",
            filtered_reproj_errors,
            plot_type=GtsfmMetric.PlotType.BOX,
        ),
    }
    return other_pipeline_metrics


def save_other_pipelines_metrics(
    colmap_format_outputs: Dict[str, str],
    gtsfm_json_dirpath: str,
    gtsfm_metric_filenames: List[str],
    reproj_error_threshold: int,
) -> None:
    """Converts the outputs of other SfM pipelines to GTSfMMetricsGroups saved as json files.

    Creates folders for each additional SfM pipeline that contain GTSfMMetricsGroups (stored as json files)
    containing the same metrics as GTSFM_MODULE_METRICS_FNAMES (as defined in plot_metrics.py).
    If one of the GTSfM metrics is not available from another SfM pipeline, then
    the metric is left blank for that pipeline.

    Args:
        colmap_format_outputs: a Dict of paths to directories containing outputs of other SfM pipelines
          in COLMAP format i.e. cameras.txt, images.txt, and points3D.txt files.
        gtsfm_json_dirpath: Path to folder that contains metrics as json files.
        gtsfm_metric_filenames: List of filenames of metrics that are produced by GTSfM.
        reproj_error_threshold: Reprojection error threshold for filtering tracks.
    """
    for other_pipeline_name, colmap_format_output_dirpath in colmap_format_outputs.items():
        cameras, images, points3d = colmap_io.read_model(path=colmap_format_output_dirpath, ext=".txt")
        other_pipeline_metrics = compute_metrics_from_txt(cameras, images, points3d, reproj_error_threshold)

        # Create json files of GTSfM Metrics for other pipelines that are comparable to GTSfM's result_metric directory
        for filename in gtsfm_metric_filenames:
            other_pipeline_group_metrics = []
            gtsfm_metrics_group = GtsfmMetricsGroup.parse_from_json(os.path.join(gtsfm_json_dirpath, filename))
            for gtsfm_metric in gtsfm_metrics_group.metrics:
                if gtsfm_metric.name in other_pipeline_metrics.keys():
                    other_pipeline_group_metrics.append(other_pipeline_metrics[gtsfm_metric.name])
            other_pipeline_new_metrics_group = GtsfmMetricsGroup(gtsfm_metrics_group.name, other_pipeline_group_metrics)
            os.makedirs(os.path.join(gtsfm_json_dirpath, other_pipeline_name), exist_ok=True)
            other_pipeline_new_metrics_group.save_to_json(
                os.path.join(gtsfm_json_dirpath, other_pipeline_name, os.path.basename(filename))
            )
