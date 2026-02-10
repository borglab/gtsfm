"""Plot pose AUC boxplots across a result tree.

This script scans a result root directory for a list of subdirectories.
The first subdirectory is used for pre-BA metrics; all listed subdirectories are
used for post-BA metrics. Optionally includes an extra GT folder name.

It extracts pose AUC values (pose_auc_@X_deg) from BA metrics and plots box plots per threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import plotly.graph_objects as go  # type: ignore

import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()


def _find_pose_auc(data: Any) -> Dict[float, float]:
    """Recursively find pose AUCs in nested dicts."""
    aucs: Dict[float, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(key, str) and key.startswith("pose_auc_@") and key.endswith("_deg"):
                try:
                    threshold_str = key[len("pose_auc_@") : -len("_deg")]
                    threshold = float(threshold_str)
                    aucs[threshold] = float(value)
                except (ValueError, TypeError):
                    continue
            else:
                aucs.update(_find_pose_auc(value))
    elif isinstance(data, list):
        for item in data:
            aucs.update(_find_pose_auc(item))
    return aucs


def _load_pose_auc(json_path: Path) -> Dict[float, float]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", json_path, exc)
        return {}
    return _find_pose_auc(data)


def _find_metric_values(data: Any, key_name: str) -> list[float]:
    """Recursively find scalar values for a given metric key."""
    values: list[float] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == key_name:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
            else:
                values.extend(_find_metric_values(value, key_name))
    elif isinstance(data, list):
        for item in data:
            values.extend(_find_metric_values(item, key_name))
    return values


def _load_metric_values(json_path: Path, key_name: str) -> list[float]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", json_path, exc)
        return []
    return _find_metric_values(data, key_name)


def _short_label_from_path(rel_path: str) -> str:
    parts = Path(rel_path).parts
    for part in parts:
        if part.startswith("C_"):
            return part
    return "root"


def _collect_pre_metrics(result_root: Path, subdir: str) -> list[Dict[float, float]]:
    values: list[Dict[float, float]] = []
    paths = list(result_root.rglob(f"{subdir}/metrics/ba_metrics_pre_ba.json"))
    logger.info("Pre-BA: found %d files for %s", len(paths), subdir)
    for pre_path in paths:
        aucs = _load_pose_auc(pre_path)
        if aucs:
            values.append(aucs)
    return values


def _collect_post_metrics(result_root: Path, subdir: str) -> list[Dict[float, float]]:
    values: list[Dict[float, float]] = []
    metrics_dirs = list(result_root.rglob(f"{subdir}/metrics"))
    logger.info("Post-BA: found %d metrics dirs for %s", len(metrics_dirs), subdir)
    for metrics_dir in metrics_dirs:
        candidate_paths = [
            metrics_dir / "ba_metrics_post_ba.json",
            metrics_dir / "ba_metrics.json",
        ]
        metric_path = next((p for p in candidate_paths if p.exists()), None)
        if metric_path is None:
            continue
        aucs = _load_pose_auc(metric_path)
        if aucs:
            values.append(aucs)
    return values


def _collect_post_final_errors(result_root: Path, subdir: str) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for metrics_dir in result_root.rglob(f"{subdir}/metrics"):
        candidate_paths = [
            metrics_dir / "ba_metrics_post_ba.json",
            metrics_dir / "ba_metrics.json",
        ]
        metric_path = next((p for p in candidate_paths if p.exists()), None)
        if metric_path is None:
            continue
        rel_path = str(metric_path.relative_to(result_root))
        label = _short_label_from_path(rel_path)
        for value in _load_metric_values(metric_path, "final_ba_error"):
            values.append((label, value))
    return values


def _collect_thresholds(*groups: Iterable[Dict[float, float]]) -> list[float]:
    thresholds: set[float] = set()
    for group in groups:
        for aucs in group:
            thresholds.update(aucs.keys())
    return sorted(thresholds)


def _build_box_trace(
    label: str,
    values: list[Dict[float, float]],
    thresholds: list[float],
) -> go.Box | None:
    x_vals: list[str] = []
    y_vals: list[float] = []
    for threshold in thresholds:
        for aucs in values:
            if threshold in aucs:
                x_vals.append(str(threshold))
                y_vals.append(aucs[threshold])
    if not y_vals:
        return None
    return go.Box(
        x=x_vals,
        y=y_vals,
        name=label,
        boxmean="sd",
    )


def _build_scatter_trace(label: str, values: list[tuple[str, float]]) -> go.Scatter | None:
    if not values:
        return None
    x_vals = [item[0] for item in values]
    y_vals = [item[1] for item in values]
    return go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers",
        name=label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", required=True, help="Root directory to scan for metrics.")
    parser.add_argument(
        "--subdirs",
        nargs="+",
        required=True,
        help="List of subdirectories; first is pre-BA, all are post-BA.",
    )
    parser.add_argument(
        "--gt_folder_name",
        default=None,
        help="Optional extra post-BA subdir (e.g., ba_robust_gt_calibfix).",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output HTML path (default: <result_root>/metrics/pose_auc_boxplot.html).",
    )
    args = parser.parse_args()

    result_root = Path(args.result_root)
    if not result_root.exists():
        raise FileNotFoundError(f"Result root not found: {result_root}")

    subdirs = list(args.subdirs)
    pre_subdir = subdirs[0]
    post_subdirs = list(subdirs)
    if args.gt_folder_name:
        post_subdirs.append(args.gt_folder_name)

    pre_vals = _collect_pre_metrics(result_root, pre_subdir)
    post_vals_by_label: dict[str, list[Dict[float, float]]] = {}
    post_final_by_label: dict[str, list[tuple[str, float]]] = {}
    for subdir in post_subdirs:
        post_vals_by_label[subdir] = _collect_post_metrics(result_root, subdir)
        post_final_by_label[subdir] = _collect_post_final_errors(result_root, subdir)
        logger.info(
            "Post-BA AUCs for %s: %d entries, final_ba_error entries: %d",
            subdir,
            len(post_vals_by_label[subdir]),
            len(post_final_by_label[subdir]),
        )

    thresholds = _collect_thresholds(pre_vals, *post_vals_by_label.values())
    if not thresholds:
        logger.warning("No pose AUCs found under %s", result_root)
        return

    fig = go.Figure()
    pre_trace = _build_box_trace("pre_ba", pre_vals, thresholds)
    if pre_trace is not None:
        fig.add_trace(pre_trace)
    for label, values in post_vals_by_label.items():
        post_label = f"{label}_post_ba"
        post_trace = _build_box_trace(post_label, values, thresholds)
        if post_trace is not None:
            fig.add_trace(post_trace)
        else:
            example_keys = list(values[0].keys()) if values else []
            logger.warning(
                "No post-BA trace for %s (entries=%d, example thresholds=%s)",
                label,
                len(values),
                example_keys[:5],
            )

    fig.update_layout(
        title="Pose AUCs across result tree",
        xaxis_title="Pose AUC threshold (deg)",
        yaxis_title="AUC",
        boxmode="group",
    )

    output_file = args.output_file
    if output_file is None:
        output_file = str(result_root / "metrics" / "pose_auc_boxplot.html")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info("Saved plot to %s", output_path)

    if any(post_final_by_label.values()):
        fig_cost = go.Figure()
        for label, values in post_final_by_label.items():
            post_label = f"{label}_post_ba"
            post_cost_trace = _build_scatter_trace(post_label, values)
            if post_cost_trace is not None:
                fig_cost.add_trace(post_cost_trace)
        fig_cost.update_layout(
            title="Final BA error per file",
            xaxis_title="Cluster",
            yaxis_title="Final BA error",
            xaxis_tickangle=45,
        )
        cost_path = output_path.parent / "final_ba_error_boxplot.html"
        fig_cost.write_html(str(cost_path))
        logger.info("Saved final BA error plot to %s", cost_path)
    else:
        logger.warning("No final_ba_error values found; skipping cost plot.")


if __name__ == "__main__":
    main()
