"""
Use Plotly to create a dashboard that compares the metrics across all the benchmarks from the CI.

The dashboard is a heatmap representing a 2d table, with text annotations added to it.

Authors: John Lambert
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import plotly.graph_objects as go
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from plotly.graph_objs.layout import Annotation, Font, Margin, XAxis, YAxis

import gtsfm.evaluation.merge_reports as report_utils
import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.utils.metrics as metrics_utils


HEATMAP_WIDTH = 1500
HEATMAP_HEIGHT = 900
NUM_COLORS_COLORMAP = 100
MAX_NUM_CHARS_ARTIFACT_FNAME = 35

MIN_RENDERABLE_PERCENT_CHANGE = -20
MAX_RENDERABLE_PERCENT_CHANGE = 20

DASHBOARD_HTML_SAVE_FPATH = Path(__file__).parent.parent.parent / "visual_comparison_dashboard.html"

ZIP_FNAMES = [
    "deep_front_end-2011205_rc3-20-png-wget-astronet-1024-true.zip",
    "deep_front_end-door-12-12-JPG-test_data-olsson-loader-1296-true.zip",
    "deep_front_end-notre-dame-20-20-jpg-gdrive-colmap-loader-760-false.zip",
    "deep_front_end-skydio-8-8-jpg-gdrive-colmap-loader-760-true.zip",
    "deep_front_end-skydio-32-32-jpg-gdrive-colmap-loader-760-true.zip",
    "sift_front_end-2011205_rc3-65-png-wget-astronet-1024-true.zip",
    "sift_front_end-door-12-12-JPG-test_data-olsson-loader-1296-true.zip",
    "sift_front_end-palace-fine-arts-281-25-jpg-wget-olsson-loader-320-true.zip",
    "sift_front_end-skydio-8-8-jpg-gdrive-colmap-loader-760-true.zip",
    "sift_front_end-skydio-32-32-jpg-gdrive-colmap-loader-760-true.zip",
]

TABLE_NAMES = [
    "Verifier Summary",
    "Inlier Support Processor Summary",
    "Rotation Cycle Consistency Metrics",
    "Cycle Consistent Frontend Summary",
    "Averaging Metrics",
    "Data Association Metrics",
    "Bundle Adjustment Metrics",
]

RED_HEX = "#df0101"
PALE_YELLOW_HEX = "#f5f6ce"
GREEN_HEX = "#31b404"


def colorscale_from_list(colorlist: List[str]) -> List[str]:
    """Create hex colorscale to interpolate between requested colors.

    Args:
        colorlist: requested colors.

    Returns:
        colorscale: list of length (NUM_COLORS_COLORMAP+1) representing a list of colors.
    """
    cmap = LinearSegmentedColormap.from_list(name="dummy_name", colors=colorlist)
    colorscale = [colors.rgb2hex(cmap(k * 1 / NUM_COLORS_COLORMAP)) for k in range(NUM_COLORS_COLORMAP + 1)]
    return colorscale


def plot_colored_table(X: List[str], Y: List[str], Z: np.ndarray) -> str:
    """Create an annotated heatmap.

    Args:
        X: labels for each column (column names).
        Y: labels for each row (row names).
        Z: 2d matrix, representing percentage changes from value for a metric on the master branch.

    Returns:
        string representing html for table.
    """

    # Clip Z to -20% and +20%. The clipping is only for the color -- the text will still display the correct numbers.
    Z_clipped = np.clip(Z, a_min=MIN_RENDERABLE_PERCENT_CHANGE, a_max=MAX_RENDERABLE_PERCENT_CHANGE)

    redgreen = [RED_HEX, PALE_YELLOW_HEX, GREEN_HEX]
    colorscale = colorscale_from_list(redgreen)
    trace = go.Heatmap(
        z=Z_clipped,
        x=X,
        y=Y,
        colorscale=colorscale,
        zmin=-MIN_RENDERABLE_PERCENT_CHANGE,
        zmax=MAX_RENDERABLE_PERCENT_CHANGE,
    )

    layout = go.Layout(
        title="Percentage Change",
        font=Font(family="Balto, sans-serif", size=12, color="rgb(68,68,68)"),
        showlegend=False,
        xaxis=XAxis(title="", showgrid=True, side="top", tickangle=-45),
        yaxis=YAxis(
            title="",
            autorange="reversed",
            showgrid=True,
        ),
        autosize=False,
        height=HEATMAP_HEIGHT,
        width=HEATMAP_WIDTH,
        margin=Margin(l=135, r=40, b=85, t=170),
    )

    fig = go.Figure(data=go.Data([trace]), layout=layout)

    annotations = go.Annotations()

    num_rows, num_cols = Z.shape
    for i in range(num_rows):
        for j in range(num_cols):
            annotations.append(
                Annotation(
                    text=str(np.round(Z[i, j], 1)) + "%",
                    x=X[j],
                    y=Y[i],
                    xref="x1",
                    yref="y1",
                    font=dict(color="rgb(25,25,25)"),
                    showarrow=False,
                )
            )
    fig["layout"].update(annotations=annotations)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def generate_dashboard(curr_master_dirpath: str, new_branch_dirpath: str) -> None:
    """Generate a dashboard showing a visual representation of the diff against master on all benchmarks.

    This script expects to find the metrics in CI artifact files and saves to the main repo directory.
    TODO(johnwlambert): read metrics from JSON, instead of from the HTML report.

    Args:
        curr_master_dirpath: path to directory containing benchmark artifacts for the master branch.
        new_branch_dirpath: path to directory containing benchmark artifacts for a new branch.
    """

    f = open(DASHBOARD_HTML_SAVE_FPATH, mode="w")

    # Write HTML headers.
    f.write("<!DOCTYPE html>" "<html>")
    f.write(metrics_report.get_html_header())

    # loop over each table in the HTML report.
    for table_name in TABLE_NAMES:

        X = []
        Y = []

        benchmark_table_vals = defaultdict(dict)

        # loop over each benchmark result (columns of table)
        for zip_fname in ZIP_FNAMES:
            # use just the first 35 chars
            X.append(zip_fname[:MAX_NUM_CHARS_ARTIFACT_FNAME])

            report1_fpath = f"{curr_master_dirpath}/results-{zip_fname}/result_metrics/gtsfm_metrics_report.html"
            tables_dict1 = report_utils.extract_tables_from_report(report1_fpath)

            report2_fpath = f"{new_branch_dirpath}/results-{zip_fname}/result_metrics/gtsfm_metrics_report.html"
            tables_dict2 = report_utils.extract_tables_from_report(report2_fpath)
            merged_tables_dict = report_utils.merge_tables(tables_dict1, tables_dict2)

            # loop over each metric within this table (rows of table)
            for i, (metric_name, master_val, branch_val) in enumerate(merged_tables_dict[table_name]):

                if branch_val is None:
                    percentage_change = np.nan
                else:
                    percentage_change = metrics_utils.compute_percentage_change(float(master_val), float(branch_val))

                if "error" in metric_name and "outlier" not in metric_name:
                    # smaller is better, so this will flip the color to green for reduced values, instead of red
                    # exception are outlier errors, which we want to get larger.
                    percentage_change *= -1
                benchmark_table_vals[metric_name][zip_fname] = percentage_change

        Z_rows = []
        for metric_name, benchmark_vals_dict in benchmark_table_vals.items():
            Z_row = []
            for zip_fname in ZIP_FNAMES:
                Z_row.append(benchmark_vals_dict.get(zip_fname, np.nan))  # default was unchanged if missing
            Z_rows.append(Z_row)
            Y.append(metric_name)

        Z = np.array(Z_rows)
        table_html = plot_colored_table(X, Y, Z)

        # Write name of the metric group in human readable form.
        f.write(metrics_report.get_html_metric_heading(table_name))
        f.write(table_html)

        # Close HTML tags.
        f.write("</html>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curr_master_dirpath",
        required=True,
        help="Path to directory containing benchmark artifacts for the master branch.",
    )
    parser.add_argument(
        "--new_branch_dirpath",
        required=True,
        help="Path to directory containing benchmark artifacts for a new branch.",
    )
    args = parser.parse_args()
    generate_dashboard(
        curr_master_dirpath=args.curr_master_dirpath,
        new_branch_dirpath=args.new_branch_dirpath,
    )
