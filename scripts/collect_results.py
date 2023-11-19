"""Aggregate into a Markdown or LaTeX table results of experiments comparing various front-ends.

We use the `tabulate` package to print the table to STDOUT.
"""

import argparse
import datetime
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tabulate import tabulate

import gtsfm.utils.io as io_utils


isp_fname = "verifier_summary_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json"
isp_metrics = [
    "rot3_angular_errors_deg",
    "trans_angular_errors_deg",
    "pose_errors_deg",
    "total_correspondence_generation_duration_sec",
    "total_two_view_estimation_duration_sec",
]

retriever_fname = "retriever_metrics.json"
retriever_metrics = ["num_input_images", "num_retrieved_image_pairs", "retriever_duration_sec"]

vg_fname = "view_graph_estimation_metrics.json"
vg_metrics = [
    "num_input_measurements",
    "num_inlier_measurements",
    "num_outlier_measurements",
    "inlier_R_angular_errors_deg",
    "inlier_U_angular_errors_deg",
]

ra_fname = "rotation_averaging_metrics.json"
ra_metrics = ["rotation_angle_error_deg", "total_duration_sec"]

ta_fname = "translation_averaging_metrics.json"
ta_metrics = [
    "relative_translation_angle_error_deg",
    "translation_angle_error_deg",
    "total_duration_sec",
    "outlier_rejection_duration_sec",
    "optimization_duration_sec",
]

da_fname = "data_association_metrics.json"
da_metrics = ["triangulation_runtime_sec", "gtsfm_data_creation_runtime", "total_duration_sec"]

ba_result_fname = "bundle_adjustment_metrics.json"
ba_result_metrics = [
    "number_cameras",
    "number_tracks_filtered",
    "3d_track_lengths_filtered",
    "reprojection_errors_filtered_px",
    "rotation_angle_error_deg",
    "relative_translation_angle_error_deg",
    "translation_angle_error_deg",
    "pose_auc_@1_deg",
    "pose_auc_@2.5_deg",
    "pose_auc_@5_deg",
    "pose_auc_@10_deg",
    "pose_auc_@20_deg",
    "step_0_run_duration_sec",
    "step_1_run_duration_sec",
    "step_2_run_duration_sec",
    "total_run_duration_sec",
]

total_fname = "total_summary_metrics.json"
total_metrics = ["total_runtime_sec"]

# Metrics that **do not** have a median + mean value associated.
SCALAR_METRIC_NAMES = [
    "number_cameras",
    "number_tracks_filtered",
    "num_input_images",
    "num_retrieved_image_pairs",
    "pose_auc_@1_deg",
    "pose_auc_@2.5_deg",
    "pose_auc_@5_deg",
    "pose_auc_@10_deg",
    "pose_auc_@20_deg",
    "step_0_run_duration_sec",
    "step_1_run_duration_sec",
    "step_2_run_duration_sec",
    "total_run_duration_sec",
    "total_duration_sec",
    "outlier_rejection_duration_sec",
    "optimization_duration_sec",
    "num_input_measurements",
    "num_inlier_measurements",
    "num_outlier_measurements",
    "total_correspondence_generation_duration_sec",
    "total_two_view_estimation_duration_sec",
    "triangulation_runtime_sec",
    "gtsfm_data_creation_runtime",
    "total_runtime_sec",
    "retriever_duration_sec",
]

SECTION_FILE_NAMES = [retriever_fname, isp_fname, vg_fname, ra_fname, ta_fname, da_fname, ba_result_fname, total_fname]
SECTION_METRIC_LISTS = [
    retriever_metrics,
    isp_metrics,
    vg_metrics,
    ra_metrics,
    ta_metrics,
    da_metrics,
    ba_result_metrics,
    total_metrics,
]
SECTION_NICKNAMES = ["retriever", "isp", "vg", "ra", "ta", "da", "ba", "total"]


def main(experiment_roots: Sequence[Path], output_fpath: str) -> None:
    """ """
    # Store each column as mappings of (key, value) pairs, where (metric_name, experiment_value).
    table = defaultdict(list)
    headers = ["method_name"]

    method_idx = 0
    for experiment_root in experiment_roots:
        dirpath = Path(experiment_root) / "result_metrics"
        frontend_name = Path(experiment_root).name
        table["method_name"].append(frontend_name)

        for json_fname, metric_names, nickname in zip(
            SECTION_FILE_NAMES,
            SECTION_METRIC_LISTS,
            SECTION_NICKNAMES,
        ):
            section_name = Path(json_fname).stem
            print(f"{dirpath}/{json_fname}")
            json_data = io_utils.read_json_file(f"{dirpath}/{json_fname}")[section_name]
            for metric_name in metric_names:
                full_metric_name = f"{nickname}_" + " ".join(metric_name.split("_"))
                if method_idx == 0:
                    headers.append(full_metric_name)

                if "pose_auc_" in metric_name and metric_name in SCALAR_METRIC_NAMES:
                    table[full_metric_name].append(json_data[metric_name] * 100)
                elif metric_name in SCALAR_METRIC_NAMES:
                    print(f"{metric_name}: {json_data[metric_name]}")
                    table[full_metric_name].append(json_data[metric_name])
                else:
                    med = f"{json_data[metric_name]['summary']['median']:.1f}"
                    mean = f"{json_data[metric_name]['summary']['mean']:.1f}"
                    print(f"Med / Median {metric_name}: {med} / {mean}")
                    table[full_metric_name].append(f"{med} / {mean}")
        method_idx += 1

    # We treat the defaultdict as a table (dict of iterables).
    stdout_lines = tabulate(table, headers, tablefmt="fancy_grid")
    print(stdout_lines)
    save_table_to_tsv(table, headers, output_fpath)


def _make_runtime_pie_chart(experiment_roots: Sequence[Path]) -> None:
    """Make pie chart to depict runtime breakdown for each run."""
    for experiment_root in experiment_roots:
        runtime_labels = []
        runtimes = []

        runtimes_sum = 0.0
        total_runtime = 0.0

        dirpath = Path(experiment_root) / "result_metrics"
        for json_fname, metric_names, nickname in zip(
            SECTION_FILE_NAMES,
            SECTION_METRIC_LISTS,
            SECTION_NICKNAMES,
        ):
            section_name = Path(json_fname).stem
            json_data = io_utils.read_json_file(f"{dirpath}/{json_fname}")[section_name]
            for metric_name in metric_names:
                full_metric_name = f"{nickname}_" + " ".join(metric_name.split("_"))
                if "sec" not in metric_name:
                    continue

                runtime = json_data[metric_name]
                if full_metric_name == "total_total runtime sec":
                    total_runtime = runtime
                elif full_metric_name in [
                    "ta_total duration sec",
                    "da_total duration sec",
                    "ba_total run duration sec",
                ]:
                    # Section components are measured individually, so ignore section runtime.
                    pass
                else:
                    runtimes_sum += runtime
                    runtimes.append(runtime)
                    runtime_labels.append(full_metric_name)

        # Compute and plot unmeasured component.
        remainder_runtime = total_runtime - runtimes_sum
        runtime_labels.append("remainder_sec")
        runtimes.append(remainder_runtime)

        # Create uniform purple to yellow colormap to prevent color re-use in pie chart.
        n_colors = len(runtimes)
        cs = cm.viridis(np.arange(n_colors) / n_colors * 1.0)

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.pie(runtimes, labels=runtime_labels, autopct="%1.1f%%", textprops={"fontsize": 10}, colors=cs)
        plt.title("Runtime Breakdown for " + str(experiment_root.name))
        plt.show()


def save_table_to_tsv(table: DefaultDict, headers: Sequence[str], output_fpath: str) -> None:
    """Save a table to tsv, with given column headers."""
    content = tabulate(table, headers, tablefmt="tsv")
    text_file = open(output_fpath, "w")
    text_file.write(content)
    text_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_fpath",
        type=str,
        default=f"./{datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')}_aggregated_results_table.tsv",
        help="Output file path where spreadsheet (tsv) containing aggregated results will be written.",
    )
    parser.add_argument(
        "--user_root",
        type=str,
        required=True,
        help="Root directory where experiment results are stored, with a subdirectory for each experiment.",
    )
    parser.add_argument(
        "--show_runtime_pie_chart",
        action="store_true",
        help="Whether to plot runtime breakdown for each experiment as a pie chart (defaults to not display).",
    )
    args = parser.parse_args()

    user_root = Path(args.user_root)
    if not user_root.exists():
        raise FileNotFoundError(f"No directory was found at {user_root}.")

    if Path(args.output_fpath).suffix != ".tsv":
        raise ValueError("Output file path must end in `.tsv`")

    experiment_roots = sorted(list(user_root.glob("*")))
    experiment_roots = [d for d in experiment_roots if d.is_dir()]

    main(experiment_roots=experiment_roots, output_fpath=args.output_fpath)
    if args.show_runtime_pie_chart:
        _make_runtime_pie_chart(experiment_roots=experiment_roots)
