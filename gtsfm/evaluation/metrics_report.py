"""Functions to generate a report of metrics with tables and plots.

A HTML report can be generated using the generate_metrics_report_html() function, 
if called with a list of GtsfmMetricsGroup. 
The HTML report has headers, section headings, tables generated using tabulate
and grids of box or histogram plots generated using plotly.

Authors: Akshay Krishnan
"""
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as psubplot
from tabulate import tabulate

import gtsfm.evaluation.metrics as metrics
from gtsfm.evaluation.metrics import GtsfmMetricsGroup

SUBPLOTS_PER_ROW = 3


def get_readable_metric_name(metric_name: str) -> str:
    """Helper to convert a metric name separated by underscores to readable format.

    In readable format, each word is capitalized and are separated by spaces.
    Ex: bundle_adjustment_metrics -> Bundle Adjustment Metrics

    Args:
        metric_name: where words are separated by underscores.

    Returns:
        readable metric name where words are separated by spaces.
    """
    words = metric_name.split("_")
    words = [word.capitalize() for word in words]
    return " ".join(words)


def create_table_for_scalar_metrics(metrics_dict: Dict[str, Union[float, int]]) -> str:
    """Creates a table in HTML format for scalar metrics.

    Args:
        metrics_dict: A dict, where keys are names of metrics and values are
        the dictionary representation of the metric.

    Returns:
        Table with scalar metrics and their values in HTML format.
    """
    table = {"Metric name": list(metrics_dict.keys()), "Value": list(metrics_dict.values())}
    return tabulate(table, headers="keys", tablefmt="html")


def create_table_for_scalar_metrics_and_compare(metrics_dict: Dict[str, List[Union[float, int]]]) -> str:
    """Creates a table in HTML format with an additional column for scalar metrics from COLMAP.

    Args:
        metrics_dict: A dict, where keys are names of metrics and values are
        a list of metric values from each pipeline.

    Returns:
        Table with scalar metrics and their values in HTML format.
    """
    for metric_key, metric_values in metrics_dict.items():
        for metric_index, metric_value in enumerate(metric_values):
            if isinstance(metric_value, float):
                if metric_value.is_integer():
                    metric_values[metric_index] = int(metric_value)
                else:
                    metric_values[metric_index] = round(metric_value, 3)
    table = {
        "Metric name": list(metrics_dict.keys()),
        "GTSfM": list(item[0] for item in metrics_dict.values()),
        "COLMAP": list(item[1] for item in metrics_dict.values()),
    }
    return tabulate(table, headers="keys", tablefmt="html")


def create_plots_for_distributions(metrics_dict: Dict[str, Any]) -> str:
    """Creates plots for 1D distribution metrics using plotly, and converts them to HTML.

    The plots are arranged in a grid, with each row having SUBPLOTS_PER_ROW (3) columns.
    For a certain metric, these can be either histogram or box according to the metric's property.

    Args:
        metrics_dict: A dict, where keys are names of metrics and values are
        the dictionary representation of the metric.
    Returns:
        Plots in a grid converted to HTML as a string.
    """
    distribution_metrics = []
    # Separate all the 1D distribution metrics.
    for metric, value in metrics_dict.items():
        if isinstance(value, dict):
            all_nan_summary = all(np.isnan(v) for v in value[metrics.SUMMARY_KEY].values())
            if not all_nan_summary:
                distribution_metrics.append(metric)
    if len(distribution_metrics) == 0:
        return ""

    # Setup figure layout - number of rows and columns.
    num_rows = (len(distribution_metrics) + SUBPLOTS_PER_ROW - 1) // SUBPLOTS_PER_ROW
    fig = psubplot.make_subplots(rows=num_rows, cols=SUBPLOTS_PER_ROW, subplot_titles=distribution_metrics)
    fig.update_layout({"height": 512 * num_rows, "width": 1024, "showlegend": False})
    i = 0

    # Iterate over all metrics.
    for metric_name, metric_value in metrics_dict.items():
        # Check if this is a 1D distribution metric and has a summary.
        if metric_name not in distribution_metrics or metrics.SUMMARY_KEY not in metric_value:
            continue
        row = i // SUBPLOTS_PER_ROW + 1
        col = i % SUBPLOTS_PER_ROW + 1
        i += 1
        # Histogram metrics are plotted directly from summary.
        if "histogram" in metric_value[metrics.SUMMARY_KEY]:
            histogram = metric_value[metrics.SUMMARY_KEY]["histogram"]
            fig.add_trace(
                go.Bar(x=list(histogram.keys()), y=list(histogram.values()), name=metric_name), row=row, col=col
            )
        elif "quartiles" in metric_value[metrics.SUMMARY_KEY]:
            # If all values are available, use them to create box plot.
            if metrics.FULL_DATA_KEY in metric_value:
                fig.add_trace(go.Box(y=metric_value[metrics.FULL_DATA_KEY], name=metric_name), row=row, col=col)
            # Else use summary to create box plot.
            else:
                quartiles = metric_value[metrics.SUMMARY_KEY]["quartiles"]
                fig.add_trace(
                    go.Box(
                        q1=[quartiles["q1"]],
                        median=[quartiles["q2"]],
                        q3=[quartiles["q3"]],
                        lowerfence=[quartiles["q0"]],
                        upperfence=[quartiles["q4"]],
                        name=metric_name,
                    ),
                    row=row,
                    col=col,
                )

    # Return the figure converted to HTML.
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def get_figures_for_metrics(metrics_group: GtsfmMetricsGroup) -> Tuple[str, str]:
    """Gets the tables and plots for individual metrics in a metrics group.

    All scalar metrics are reported in the table.
    Metrics of 1-D distributions have an entry in the table for the mean,
    and a histogram or box plot as per their property.

    Args:
        metrics_group: A GtsfmMetricsGroup for any gtsfm module.

    Returns:
        A tuple of table and plotly figures as HTML code.
    """
    scalar_metrics = {}
    metrics_dict = metrics_group.get_metrics_as_dict()[metrics_group.name]
    # Separate the scalar metrics.
    for metric_name, value in metrics_dict.items():
        if isinstance(value, dict):
            # Metrics with a dict representation must contain a summary.
            if metrics.SUMMARY_KEY not in value:
                raise ValueError(f"Metric {metric_name} does not contain a summary.")
            # Add a scalar metric for median of 1D distributions.
            scalar_metrics["median_" + metric_name] = value[metrics.SUMMARY_KEY]["median"]
        else:
            scalar_metrics[metric_name] = value
    table = create_table_for_scalar_metrics(scalar_metrics)
    plots_fig = create_plots_for_distributions(metrics_dict)
    return table, plots_fig


def add_summary(metric_name, metric_value, scalar_metrics):
    # Metrics with a dict representation must contain a summary.
    if metrics.SUMMARY_KEY not in metric_value:
        raise ValueError(f"Metric {metric_name} does not contain a summary.")
    # Add a scalar metric for median of 1D distributions.
    median_nan = metric_value[metrics.SUMMARY_KEY]["median"] != metric_value[metrics.SUMMARY_KEY][
        "median"]
    if median_nan:
        scalar_metrics["median_" + metric_name].append("")
    else:
        scalar_metrics["median_" + metric_name].append(metric_value[metrics.SUMMARY_KEY]["median"])

def add_metric(gtsfm_metric_name, gtsfm_metric_value, scalar_metrics, other_pipelines_metric_dicts):
    """
    Args:
        gtsfm_metric_name: Use GTSfM metric name for adding metrics (other pipelines share metric names)
        gtsfm_metric_value: GTSfM metric value
        scalar_metrics: Dictionary with metric names as keys and a list of metric values for each pipeline
        other_pipelines_metric_dicts: Dictionaries of metric name, metric value key-value pairs for each other pipeline

    """
    other_pipelines_metric_values = []
    for other_pipeline_metric_dict in other_pipelines_metric_dicts:
        if gtsfm_metric_name in other_pipeline_metric_dict:
            other_pipelines_metric_values.append(other_pipeline_metric_dict[gtsfm_metric_name])
        else:
            other_pipelines_metric_values.append("")

    if isinstance(gtsfm_metric_value, dict):
        add_summary(gtsfm_metric_name, gtsfm_metric_value, scalar_metrics)
        for other_pipeline_metric_value in other_pipelines_metric_values:
            if isinstance(other_pipeline_metric_value, dict):
                add_summary(gtsfm_metric_name, other_pipeline_metric_value, scalar_metrics)
            else:
                other_pipeline_metric_value = {'summary': {'median': ""}}
                add_summary(gtsfm_metric_name, other_pipeline_metric_value, scalar_metrics)
    else:
        scalar_metrics[gtsfm_metric_name].append(gtsfm_metric_value)
        for other_pipeline_metric_value in other_pipelines_metric_values:
            scalar_metrics[gtsfm_metric_name].append(other_pipeline_metric_value)

def get_figures_for_metrics_and_compare(
    metrics_group: GtsfmMetricsGroup, colmap_metrics_group: GtsfmMetricsGroup
) -> Tuple[str, str]:
    """Gets the tables and plots for individual metrics in a metrics group.

    All scalar metrics are reported in the table.
    Metrics of 1-D distributions have an entry in the table for the median,
    and a histogram or box plot as per their property.

    Args:
        metrics_group: A GtsfmMetricsGroup for any gtsfm module.
        colmap_metrics_group: A GtsfmMetricsGroup for corresponding colmap modules.

    Returns:
        A tuple of table and plotly figures as HTML code.
    """
    other_pipeline_metrics_groups = [colmap_metrics_group]
    scalar_metrics = defaultdict(list)

    gtsfm_metric_dict = metrics_group.get_metrics_as_dict()[metrics_group.name]
    other_pipelines_metric_dicts = []
    for other_pipeline_metrics_group in other_pipeline_metrics_groups:
        other_pipeline_dict = other_pipeline_metrics_group.get_metrics_as_dict()[other_pipeline_metrics_group.name]
        other_pipelines_metric_dicts.append(other_pipeline_dict)

    for gtsfm_metric_name, gtsfm_metric_value in gtsfm_metric_dict.items():
        add_metric(gtsfm_metric_name, gtsfm_metric_value, scalar_metrics, other_pipelines_metric_dicts)

    table = create_table_for_scalar_metrics_and_compare(scalar_metrics)

    plots_fig = ""
    all_metrics_groups = [metrics_group] + other_pipeline_metrics_groups
    for metrics_group in all_metrics_groups:
        plots_fig += create_plots_for_distributions(metrics_group.get_metrics_as_dict()[metrics_group.name])
    return table, plots_fig


def get_html_metric_heading(metric_name: str) -> str:
    """Helper to get the HTML heading for a metric name.

    This converts a "metric_name" to "Metric Name" (for readability) and adds some style.

    Returns:
        HTML for heading as a string.
    """
    metric_name = get_readable_metric_name(metric_name)
    metric_html = f'<p style="font-size:25px;font-family:Arial">{metric_name}</p>'
    return metric_html


def get_html_header() -> str:
    """Helper to get a HTML header with some CSS styles for tables.""

    Returns:
        A HTML header as string.
    """
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


def generate_metrics_report_html(
    metrics_groups: List[GtsfmMetricsGroup],
    html_path: str,
    colmap_metrics_groups,
    metric_paths: List[str],
) -> None:
    """Generates a report for metrics groups with plots and tables and saves it to HTML.

    Args:
        metrics_groups: List of metrics to be reported.
        html_path: Path where this report is written to (default: output_dir/gtsfm_metrics_report.html).
        colmap_output_dir: Optional; If a path to a directory containing a COLMAP reconstruction
          (as cameras.txt, images.txt, and points3D.txt) is provided, the COLMAP metrics will also be
          included in the report.
        metric_paths: A list of paths to GTSfM metrics.
    """
    with open(html_path, mode="w") as f:
        # Write HTML headers.
        f.write("<!DOCTYPE html>" "<html>")
        f.write(get_html_header())

        # Iterate over all metrics groups
        for i, metrics_group in enumerate(metrics_groups):

            # Write name of the metric group in human readable form.
            f.write(get_html_metric_heading(metrics_group.name))

            # Write plots and tables.
            if colmap_metrics_groups is None:
                table, plots_fig = get_figures_for_metrics(metrics_group)
            else:
                table, plots_fig = get_figures_for_metrics_and_compare(metrics_group, colmap_metrics_groups[i])
            f.write(table)
            if plots_fig is not None:
                f.write(plots_fig)

        # Close HTML tags.
        f.write("</html>")
