"""Script to generate a report of metrics with tables and plots 
using the metrics that have been logged as JSON in a previous run of the pipeline. 

Authors: Akshay Krishnan
"""
import argparse
import os
from pathlib import Path
from typing import Optional

import gtsfm.evaluation.compare_metrics as compare_metrics
import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.utils.logger as logger_utils
from gtsfm.evaluation.metrics import GtsfmMetricsGroup

logger = logger_utils.get_logger()

GTSFM_MODULE_METRICS_FNAMES = [
    "frontend_summary.json",
    "view_graph_estimation_metrics.json"
    "rotation_averaging_metrics.json",
    "translation_averaging_metrics.json",
    "data_association_metrics.json",
    "bundle_adjustment_metrics.json",
    "total_summary_metrics.json"
]


def save_other_metrics(other_pipeline_files_dirpath: str, other_pipeline_json_path: str, reproj_error_threshold: int):
    """Saves other metrics as GTSfM Metrics Groups in json files.

    Args:
        other_pipeline_files_dirpath: The path to a directory containing another SfM pipeline's as txt files.
        other_pipeline_json_path: The path to the directory where another SfM pipeline's
          output will be saved in json files.
        reproj_error_threshold: reprojection error threshold for filtering tracks.

    """
    if Path(other_pipeline_files_dirpath).exists():
        txt_metric_paths = {
            os.path.basename(other_pipeline_json_path): other_pipeline_files_dirpath,
        }
        json_path = os.path.dirname(other_pipeline_json_path)
        compare_metrics.save_other_pipelines_metrics(
            txt_metric_paths, json_path, GTSFM_MODULE_METRICS_FNAMES, reproj_error_threshold
        )
    else:
        logger.info("%s does not exist", other_pipeline_files_dirpath)


def create_metrics_plots_html(
    json_path: str, output_dir: str, colmap_json_dirpath: Optional[str], openmvg_json_dirpath: Optional[str]
) -> None:
    """Creates a HTML report of metrics from frontend, averaging, data association and bundle adjustment.

    Reads the metrics from JSON files in a previous run.

    Args:
        json_path: Path to folder that contains GTSfM metrics as json files.
        output_dir: directory to save the report, uses json_path if empty.
        colmap_json_dirpath: The path to the directory of colmap outputs in json files.
        openmvg_json_dirpath: The path to the directory of openmvg outputs in json files.
    """
    gtsfm_metrics_groups = []
    # The provided JSON path must contain these files which contain metrics from the respective modules.

    metric_paths = []
    for filename in GTSFM_MODULE_METRICS_FNAMES:
        logger.info("Adding metrics from %s", filename)
        metric_path = os.path.join(json_path, filename)
        metric_paths.append(metric_path)
        gtsfm_metrics_groups.append(GtsfmMetricsGroup.parse_from_json(metric_path))
    if len(output_dir) == 0:
        output_dir = json_path
    output_file = os.path.join(output_dir, "gtsfm_metrics_report.html")
    other_pipeline_metrics_groups = {}

    colmap_metrics_groups = []
    openmvg_metrics_groups = []
    for json_path, metrics_groups, pipeline_name in zip(
        [colmap_json_dirpath, openmvg_json_dirpath],
        [colmap_metrics_groups, openmvg_metrics_groups],
        ["colmap", "openmvg"],
    ):
        if json_path is None:
            continue
        for i, metrics_group in enumerate(gtsfm_metrics_groups):
            metric_path = metric_paths[i]
            json_metric_path = os.path.join(json_path, os.path.basename(metric_path))
            metrics_groups.append(GtsfmMetricsGroup.parse_from_json(json_metric_path))
        other_pipeline_metrics_groups[pipeline_name] = metrics_groups
    metrics_report.generate_metrics_report_html(gtsfm_metrics_groups, output_file, other_pipeline_metrics_groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="result_metrics", help="Directory containing the metrics json files.")
    parser.add_argument(
        "--colmap_files_dirpath",
        default=None,
        type=str,
        help="Directory containing COLMAP output as cameras.txt, images.txt, and points3D.txt",
    )
    parser.add_argument(
        "--openmvg_files_dirpath",
        default=None,
        type=str,
        help="Directory containing OpenMVG output as cameras.txt, images.txt, and points3D.txt",
    )
    parser.add_argument("--output_dir", default="", help="Directory to save plots to. Same as metrics_dir by default.")
    parser.add_argument(
        "--reproj_error_threshold", default=3, help="Reprojection error threshold for filtering tracks."
    )
    args = parser.parse_args()
    if args.colmap_files_dirpath is not None:
        colmap_json_dirpath = os.path.join(args.metrics_dir, "colmap")
        save_other_metrics(
            args.colmap_files_dirpath, colmap_json_dirpath, args.reproj_error_threshold
        )  # saves metrics to the json path
    else:
        colmap_json_dirpath = None
    if args.openmvg_files_dirpath is not None:
        openmvg_json_dirpath = os.path.join(args.metrics_dir, "openmvg")
        save_other_metrics(
            args.openmvg_files_dirpath, openmvg_json_dirpath, args.reproj_error_threshold
        )  # saves metrics to the json path
    else:
        openmvg_json_dirpath = None

    create_metrics_plots_html(args.metrics_dir, args.output_dir, colmap_json_dirpath, openmvg_json_dirpath)
