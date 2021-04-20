"""A CLI to generate plots for saved metrics as a HTML file.

Authors: Akshay Krishnan
"""
import argparse
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psubplot
import plotly.colors as pcolors

import numpy as np
from typing import Dict, Any

MULTIVIEW_OPTIMIZER_METRICS_FILE = "multiview_optimizer_metrics.json"

COLOR0 = pcolors.DEFAULT_PLOTLY_COLORS[0]


def parse_averaging_metrics_json(json_path: str):
    """Parses the JSON file for metrics to be visualized and returns a dict from metric name to data."""
    metrics = {}
    DISTRIBUTION_KEY = "errors_list"

    # Load JSON file.
    with open(os.path.join(json_path, MULTIVIEW_OPTIMIZER_METRICS_FILE)) as f:
        metrics_dict = json.load(f)

    # Use only the distribution ignoring the stats.
    for metric_name, metric_stats in metrics_dict.items():
        metrics[metric_name] = metric_stats[DISTRIBUTION_KEY]

    return metrics

def get_all_metrics(json_path: str):
    all_metrics = {}
    all_metrics["Averaging metrics"] = parse_averaging_metrics_json(json_path)

    # TODO: parse other metrics
    return all_metrics

def get_box_plot_for_metrics(title: str, metrics: Dict[Any, Any]):
    fig = psubplot.make_subplots(rows=1, cols=len(metrics), subplot_titles=list(metrics.keys()))
    fig.update_layout({"height": 512, "width": 1024, "showlegend": False, "title": title})  # pixels
    for i, (metric_name, metric_value) in enumerate(metrics.items()):
        fig.add_trace(go.Box(y=metric_value, name=metric_name, marker={"color": COLOR0}), row=1, col=i+1)
    return fig       

def create_metrics_plots_html(json_path: str, output_dir: str):
    all_metrics = get_all_metrics(json_path)
    with open(os.path.join(output_dir, 'gtsfm_metrics_plots.html'), mode='a') as f:
        for metrics_group_name, metrics in all_metrics.items():
            fig = get_box_plot_for_metrics(metrics_group_name, metrics)
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="result_metrics", help="Directory containing the metrics json files.")
    parser.add_argument("--output_dir", default="", help="Directory to save plots to. Same as metrics_dir by default.")
    args = parser.parse_args()
    create_metrics_plots_html(args.metrics_dir, args.output_dir)
