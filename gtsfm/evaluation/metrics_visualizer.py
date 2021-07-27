import os
from typing import Any, List, Dict, Union

from tabulate import tabulate
import plotly.graph_objects as go
import plotly.subplots as psubplot

from gtsfm.evaluation.metric import GtsfmMetric, GtsfmMetricsGroup

SUBPLOTS_PER_ROW = 3
SUMMARY_KEY = "summary"
DATA_KEY = "full_data"


def get_readable_metric_name(code_name: str):
    words = code_name.split("_")
    words = [word.capitalize() for word in words]
    return " ".join(words)


def create_table_for_scalar_metrics(table_name: str, metrics_dict: Dict[str, Union[float, int]]) -> go.Figure:
    table = {"Metric name": list(metrics_dict.keys()), "Value": list(metrics_dict.values())}
    return tabulate(table, headers="keys", tablefmt="html")


def create_plots_for_distributions(metrics_dict: Dict[str, Any]):
    distribution_metrics = []
    for metric, value in metrics_dict.items():
        if isinstance(value, dict):
            distribution_metrics.append(metric)
    if len(distribution_metrics) == 0:
        return None
    num_rows = (len(distribution_metrics) + SUBPLOTS_PER_ROW - 1) // SUBPLOTS_PER_ROW
    fig = psubplot.make_subplots(rows=num_rows, cols=SUBPLOTS_PER_ROW, subplot_titles=distribution_metrics)
    fig.update_layout({"height": 512 * num_rows, "width": 1024, "showlegend": False})
    i = 0
    for metric_name, metric_value in metrics_dict.items():
        if metric_name not in distribution_metrics:
            continue
        row = i // SUBPLOTS_PER_ROW + 1
        col = i % SUBPLOTS_PER_ROW + 1
        i += 1
        if DATA_KEY in metric_value:
            fig.add_trace(go.Box(y=metric_value[DATA_KEY], name=metric_name), row=row, col=col)
        elif SUMMARY_KEY in metric_value and "histogram" in metric_value[SUMMARY_KEY]:
            histogram = metric_value[SUMMARY_KEY][histogram]
            fig.add_trace(go.Bar(x=histogram.keys(), y=histogram.values(), name=metric_name), row=row, col=col)
        elif SUMMARY_KEY in metric_value and "quartiles" in metric_value[SUMMARY_KEY]:
            quartiles = metric_value[SUMMARY_KEY]["quartiles"]
            fig.add_trace(
                go.Box(
                    q1=quartiles["q1"],
                    median=quartiles["q2"],
                    q3=quartiles["q3"],
                    lowerfence=quartiles["q0"],
                    upperfence=quartiles["q4"],
                    name=metric_name,
                ),
                row=row,
                col=col,
            )
    return fig


def get_figures_for_metrics(metrics: GtsfmMetricsGroup):
    scalar_metrics = {}
    metrics_dict = metrics.get_metrics_as_dict()[metrics.name]
    for metric, value in metrics_dict.items():
        if isinstance(value, dict):
            if not SUMMARY_KEY in value:
                raise ValueError("Metric {metric} does not contain a summary.")
            scalar_metrics["mean_" + metric] = value[SUMMARY_KEY]["mean"]
            # scalar_metrics["min_" + metric] = value[SUMMARY_KEY]["min"]
            # scalar_metrics["max_" + metric] = value[SUMMARY_KEY]["max"]
        else:
            scalar_metrics[metric] = value
    table = create_table_for_scalar_metrics(metrics.name, scalar_metrics)
    plots_fig = create_plots_for_distributions(metrics_dict)
    return table, plots_fig


def get_html_metric_heading(metric_name: str):
    metric_name = get_readable_metric_name(metric_name)
    metric_html = f'<p style="font-size:25px;font-family:Arial">{metric_name}</p>'
    return metric_html


def get_html_header():
    return """<head>
                <style>
                  table {
                    font-family: arial, sans-serif;
                    border-collapse: collapse;
                    width: 768px
                  }
                  td, th {
                    border: 1px solid #999999;
                    text-align: left;
                    padding: 8px;
                  }
                  tr:nth-child(even) {
                    background-color: #dddddd;
                  }
                </style>
              </head>"""


def save_html_for_metrics_groups(metrics_groups: List[GtsfmMetricsGroup], html_path: str):
    with open(html_path, mode="w") as f:
        f.write("<!DOCTYPE html>" "<html>")
        f.write(get_html_header())
        for metrics_group in metrics_groups:
            f.write(get_html_metric_heading(metrics_group.name))
            table, plots_fig = get_figures_for_metrics(metrics_group)
            f.write(table)
            if plots_fig is not None:
                f.write(plots_fig.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write("</html>")
