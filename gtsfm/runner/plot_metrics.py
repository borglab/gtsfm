import os
import argparse

from gtsfm.evaluation.metric import GtsfmMetric, GtsfmMetricsGroup
import gtsfm.evaluation.metrics_visualizer as metrics_viz

def create_metrics_plots_html(json_path: str, output_dir: str):
    metrics_groups = []
    for filename in ['frontend_summary.json', 'multiview_optimizer_metrics.json', 'data_association_metrics.json', 'bundle_adjustment_metrics.json']:
        print('adding plot for ', filename)
        metric_path = os.path.join(json_path, filename)
        metrics_groups.append(GtsfmMetricsGroup.parse_from_json(metric_path))
    output_file = os.path.join(output_dir, 'gtsfm_metrics_report.html')
    metrics_viz.save_html_for_metrics_groups(metrics_groups, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="result_metrics", help="Directory containing the metrics json files.")
    parser.add_argument("--output_dir", default="", help="Directory to save plots to. Same as metrics_dir by default.")
    args = parser.parse_args()
    create_metrics_plots_html(args.metrics_dir, args.output_dir)
