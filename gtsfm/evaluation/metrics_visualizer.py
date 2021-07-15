import os
from typing import Any, List, Dict

import plotly.graph_objects as go
import plotly.subplots as psubplot

from gtsfm.evaluation.metric import GtsfmMetric, GtsfmMetricsGroup

SUBPLOTS_PER_ROW = 3
SUMMARY_KEY = "summary"
DATA_KEY = "full_data"

def create_table_for_scalar_metrics(table_name: str, metrics_dict: Dict[str, Union[float, int]]) -> go.Figure:
    fig = go.Figure(data=[go.Table(header=dict(values=['Metric', 'Value']), 
        cells=dict(values=[list(metrics_dict.keys()), list(metrics_dict.values())]))])
    fig.update_layout(title=table_name)
    return fig

def create_plots_for_distributions(metrics_dict: Dict[str, Any]):
    distribution_metrics = []
    for metric, value in metrics_dict.items():
        if isinstance(value, dict) and DATA_KEY in value:
            distribution_metrics.append(metric)
    if len(distribution_metrics) == 0:
        return None
    num_rows = len(distribution_metrics) / SUBPLOTS_PER_ROW
    fig = psubplot.make_subplots(rows=num_rows, cols=SUBPLOTS_PER_ROW, subplot_titles=distribution_metrics)
    int metric_count = 0
    for i, (metric_name, metric_value) in enumerate(metrics.items()):
        row = i // SUBPLOTS_PER_ROW
        col = i % SUBPLOTS_PER_ROW
        fig.add_trace(go.Box(y=metric_value[DATA_KEY], name=metric_name), row=row, col=col)
    return fig


def get_figures_for_metrics(metrics: GtsfmMetricsGroup):
    scalar_metrics = {}
    metrics_dict = metrics.get_metrics_as_dict()[metrics.name]
    for metric, value in metrics_dict.items():
        if isinstance(value, dict):
            if not SUMMARY_KEY in value:
                raise ValueError('Metric {metric} does not contain a summary.')
            scalar_metrics["mean_"+ metric] = value[SUMMARY_KEY]["mean"]
            scalar_metrics["min_"+ metric] = value[SUMMARY_KEY]["min"]
            scalar_metrics["max_"+ metric] = value[SUMMARY_KEY]["max"]
        else:
            scalar_metrics[metric] = value
    table_fig = create_table_for_scalar_metrics(metrics.name, scalar_metrics)
    plots_fig = create_plots_for_distributions(metrics_dict)
    return table_fig, plots_fig


def save_html_for_metrics_groups(metrics_groups: List[GtsfmMetricsGroup], html_path: str):
    with open(html_path, mode='a') as f:
        for metrics_group in metrics_groups:
            table_fig, plots_fig = get_figures_for_metrics(metrics_group)
            f.write(table_fig.to_html(full_html=False, include_plotlyjs='cdn'))
            if plots_fig is not None:
                f.write(table_fig.to_html(full_html=False, include_plotlyjs='cdn'))
