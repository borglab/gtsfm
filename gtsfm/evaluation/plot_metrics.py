"""Script to generate a report of metrics with tables and plots 
using the metrics that have been logged as JSON in a previous run of the pipeline. 

Authors: Akshay Krishnan
"""
import os
import argparse
from pathlib import Path


from gtsfm.evaluation.metrics import GtsfmMetricsGroup
import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.evaluation.compare_metrics as compare_metrics
import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()

# def save_colmap_metrics(colmap_files_dirpath, colmap_json_path):


def create_metrics_plots_html(json_path: str, colmap_files_dirpath, output_dir: str) -> None:
    """Creates a HTML report of metrics from frontend, averaging, data association and bundle adjustment.

    Reads the metrics from JSON files in a previous run.

    Args:
        json_path: Path to folder that contains metrics as json files.
        colmap_metrics_groups: TODO
        output_dir: directory to save the report, uses json_path if empty.
    """
    metrics_groups = []
    # The provided JSON path must contain these files which contain metrics from the respective modules.
    GTSFM_MODULE_METRICS_FNAMES = [
        "frontend_summary.json",
        "rotation_cycle_consistency_metrics.json",
        "rotation_averaging_metrics.json",
        "translation_averaging_metrics.json",
        "data_association_metrics.json",
        "bundle_adjustment_metrics.json",
    ]

    if Path(colmap_files_dirpath).exists():
        txt_metric_paths = {
            "colmap": colmap_files_dirpath,
        }
        compare_metrics.save_other_pipelines_metrics(txt_metric_paths, json_path, GTSFM_MODULE_METRICS_FNAMES)
    else:
        logger.info("%s does not exist", colmap_files_dirpath)

    metric_paths = []
    for filename in GTSFM_MODULE_METRICS_FNAMES:
        logger.info("Adding metrics from %s", filename)
        metric_path = os.path.join(json_path, filename)
        metric_paths.append(metric_path)
        metrics_groups.append(GtsfmMetricsGroup.parse_from_json(metric_path))
    if len(output_dir) == 0:
        output_dir = json_path
    output_file = os.path.join(output_dir, "gtsfm_metrics_report.html")

    colmap_metrics_groups = []
    for i, metrics_group in enumerate(metrics_groups):
        metric_path = metric_paths[i]
        colmap_metric_path = (
                metric_path[: metric_path.rindex("/")] + "/colmap" + metric_path[metric_path.rindex("/"):]
        )
        colmap_metrics_groups.append(GtsfmMetricsGroup.parse_from_json(colmap_metric_path))

    metrics_report.generate_metrics_report_html(metrics_groups, output_file, colmap_metrics_groups, metric_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="result_metrics", help="Directory containing the metrics json files.")
    parser.add_argument("--colmap_files_dirpath", default=None, type=str, help="Directory containing COLMAP output .")
    parser.add_argument("--output_dir", default="", help="Directory to save plots to. Same as metrics_dir by default.")
    args = parser.parse_args()
    # if args.colmap_files_dirpath is not None:
        # save_colmap_metrics(args.colmap_files_dirpath, colmap_json_path)  # saves metrics to the json path

    create_metrics_plots_html(args.metrics_dir, args.colmap_files_dirpath, args.output_dir)


    # create_metrics_plots_html(args.metrics_dir, args.colmap_files_dirpath, args.output_dir)
