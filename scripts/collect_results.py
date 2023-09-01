"""Aggregate into a Markdown or LaTeX table results of experiments comparing various front-ends.

We use the `tabulate` package to print the table to STDOUT.
"""

import argparse
import datetime
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Sequence

from tabulate import tabulate

import gtsfm.utils.io as io_utils


isp_fname = "verifier_summary_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json"
isp_metrics = ["rot3_angular_errors_deg", "trans_angular_errors_deg", "pose_errors_deg"]

retriever_fname = "retriever_metrics.json"
retriever_metrics = ["num_input_images", "num_retrieved_image_pairs"]

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
    "outier_rejection_duration_sec",
    "optimization_duration_sec",
]

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
    "total_duration_sec",
    "outier_rejection_duration_sec",
    "optimization_duration_sec",
    "num_input_measurements",
    "num_inlier_measurements",
    "num_outlier_measurements",
]


def main(user_root: Path, output_fpath: str) -> None:
    """ """
    # Store each column as mappings of (key, value) pairs, where (metric_name, experiment_value).
    table = defaultdict(list)
    headers = ["method_name"]

    experiment_roots = sorted(list(user_root.glob("*-*")))

    method_idx = 0
    for experiment_root in experiment_roots:

        dirpath = Path(experiment_root) / "result_metrics"
        frontend_name = Path(experiment_root).name
        table["method_name"].append(frontend_name)

        for json_fname, metric_names, nickname in zip(
            [retriever_fname, isp_fname, vg_fname, ra_fname, ta_fname, ba_result_fname],
            [retriever_metrics, isp_metrics, vg_metrics, ra_metrics, ta_metrics, ba_result_metrics],
            ["retriever", "isp", "vg", "ra", "ta", "ba"],
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
                    med = f"{json_data[metric_name]['summary']['median']:.2f}"
                    mean = f"{json_data[metric_name]['summary']['mean']:.2f}"
                    print(f"Med / Median {metric_name}: {med} / {mean}")
                    table[full_metric_name].append(f"{med} / {mean}")
        method_idx += 1

    # We treat the defaultdict as a table (dict of iterables).
    stdout_lines = tabulate(table, headers, tablefmt="fancy_grid")
    print(stdout_lines)
    save_table_to_tsv(table, headers, output_fpath)


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
    args = parser.parse_args()

    user_root = Path(args.user_root)
    if not user_root.exists():
        raise FileNotFoundError(f"No directory was found at {user_root}.")

    if Path(args.output_fpath).suffix != ".tsv":
        raise ValueError("Output file path must end in `.tsv`")

    main(user_root=user_root, output_fpath=args.output_fpath)
