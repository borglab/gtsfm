"""Script to generate a report of metrics with tables and plots 
using the metrics that have been logged as JSON in a previous run of the pipeline. 

Authors: Akshay Krishnan
"""
import os
import argparse

from gtsfm.evaluation.metrics import GtsfmMetricsGroup
import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()


def create_metrics_plots_html(json_path: str, output_dir: str) -> None:
    """Creates a HTML report of metrics from frontend, averaging, data association and bundle adjustment.

    Reads the metrics from JSON files in a previous run.

    Args:
        json_path: Path to folder that contains metrics json files.
        output_dir: directory to save the report, uses json_path if empty.
    """
    metrics_groups = []
    # The provided JSON path must contain these files which contain metrics from the respective modules.
    GTSFM_MODULE_METRICS_FNAMES = [
        "frontend_summary.json",
        "averaging_metrics.json",
        "data_association_metrics.json",
        "bundle_adjustment_metrics.json"
    ]
    for filename in GTSFM_MODULE_METRICS_FNAMES:
        logger.info("Adding metrics from %s", filename)
        metric_path = os.path.join(json_path, filename)
        metrics_groups.append(GtsfmMetricsGroup.parse_from_json(metric_path))
    if len(output_dir) == 0:
        output_dir = json_path
    output_file = os.path.join(output_dir, "gtsfm_metrics_report.html")
    metrics_report.generate_metrics_report_html(metrics_groups, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="result_metrics", help="Directory containing the metrics json files.")
    parser.add_argument("--output_dir", default="", help="Directory to save plots to. Same as metrics_dir by default.")
    args = parser.parse_args()
    create_metrics_plots_html(args.metrics_dir, args.output_dir)
