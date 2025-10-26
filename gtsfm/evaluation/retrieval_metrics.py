"""Utilities for analyzing retrieval quality against two-view geometry metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()


def save_retrieval_two_view_metrics(metrics_path: Path, plot_base_path: Path) -> None:
    """Compare NetVLAD similarity scores with pose errors after view-graph estimation.

    Args:
        metrics_path: Directory containing the two-view metrics JSON files.
        plot_base_path: Directory where scatter plots should be saved.
    """
    sim_fpath = plot_base_path / "similarity_matrix.txt"
    if not sim_fpath.exists():
        logger.warning("NetVLAD similarity matrix not found at %s. Skipping retrieval metrics.", sim_fpath)
        return

    sim = np.loadtxt(str(sim_fpath), delimiter=",")
    json_data = io_utils.read_json_file(metrics_path / "two_view_report_VIEWGRAPH_2VIEW_REPORT.json")

    sim_scores: list[float] = []
    R_errors: list[float] = []
    U_errors: list[float] = []

    for entry in json_data:
        i1 = entry["i1"]
        i2 = entry["i2"]
        R_error = entry["rotation_angular_error"]
        U_error = entry["translation_angular_error"]
        if R_error is None or U_error is None:
            continue
        sim_score = sim[i1, i2]

        sim_scores.append(float(sim_score))
        R_errors.append(R_error)
        U_errors.append(U_error)

    _save_scatter(
        x=sim_scores,
        y=R_errors,
        xlabel="Similarity score",
        ylabel="Rotation error w.r.t. GT (deg.)",
        output_path=plot_base_path / "gt_rot_error_vs_similarity_score.jpg",
    )
    _save_scatter(
        x=sim_scores,
        y=U_errors,
        xlabel="Similarity score",
        ylabel="Translation direction error w.r.t. GT (deg.)",
        output_path=plot_base_path / "gt_trans_error_vs_similarity_score.jpg",
    )
    pose_errors = np.maximum(np.array(R_errors), np.array(U_errors))
    _save_scatter(
        x=sim_scores,
        y=pose_errors.tolist(),
        xlabel="Similarity score",
        ylabel="Pose error w.r.t. GT (deg.)",
        output_path=plot_base_path / "gt_pose_error_vs_similarity_score.jpg",
    )


def _save_scatter(
    x: Sequence[float],
    y: Sequence[float],
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Helper to save a scatter plot."""
    plt.scatter(x, y, 10, color="r", marker=".")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(str(output_path), dpi=500)
    plt.close("all")
