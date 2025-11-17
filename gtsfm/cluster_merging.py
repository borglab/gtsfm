"""Utilities for merging cluster reconstructions and exporting results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from dask.distributed import Client, Future
from gtsam import Similarity3
import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils import align as align_utils
from gtsfm.utils.reprojection import compute_track_reprojection_errors
from gtsfm.utils.tree import Tree
from gtsfm.utils.tree_dask import submit_tree_map

if TYPE_CHECKING:
    from gtsfm.scene_optimizer import ClusterExecutionHandles

logger = logger_utils.get_logger()


def _compute_scene_reprojection_stats(scene: Optional[GtsfmData]) -> Optional[tuple[float, float, float, float]]:
    """Aggregate reprojection error stats for a scene."""
    if scene is None:
        return None
    cameras = scene.cameras()
    if len(cameras) == 0 or scene.number_tracks() == 0:
        return None

    error_blocks: list[np.ndarray] = []
    for track in scene.tracks():
        if track.numberMeasurements() == 0:
            continue
        errors, _ = compute_track_reprojection_errors(cameras, track)
        if errors.size == 0:
            continue
        valid_errors = errors[~np.isnan(errors)]
        if valid_errors.size > 0:
            error_blocks.append(valid_errors)

    if len(error_blocks) == 0:
        return None
    all_errors = np.concatenate(error_blocks)
    return (
        float(np.mean(all_errors)),
        float(np.median(all_errors)),
        float(np.min(all_errors)),
        float(np.max(all_errors)),
    )


def _log_scene_reprojection_stats(scene: Optional[GtsfmData], label: str) -> None:
    """Log reprojection error statistics."""
    stats = _compute_scene_reprojection_stats(scene)
    if stats is None:
        logger.info("📏 %s reprojection error stats unavailable", label)
        return
    mean, median, min_err, max_err = stats
    logger.info(
        "📏 %s reprojection error (px): mean=%.2f median=%.2f min=%.2f max=%.2f",
        label,
        mean,
        median,
        min_err,
        max_err,
    )


def _run_export_task(payload: Tuple[Optional[Path], Optional[GtsfmData]]) -> None:
    """Persist a merged reconstruction to COLMAP text format.

    Args:
        payload: Tuple pairing the directory to export into with the resolved merged reconstruction.
    """
    merged_dir, merged_scene = payload
    if merged_dir is None or merged_scene is None:
        return
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_scene.export_as_colmap_text(merged_dir)


def schedule_exports(
    client: Client, handles_tree: Tree[ClusterExecutionHandles], merged_future_tree: Tree[Future]
) -> Tree[Future]:
    """Schedule persistence of merged reconstructions for each cluster."""

    def _to_payload_with_future(
        value: tuple[ClusterExecutionHandles, Future], child_payloads: tuple[object, ...]
    ) -> tuple[Optional[Path], Future]:
        handle, merged_future = value
        is_leaf = len(child_payloads) == 0
        merged_dir = None if is_leaf else handle.output_paths.results / "merged"
        return merged_dir, merged_future

    export_payload_tree = Tree.zip(handles_tree, merged_future_tree).map_with_children(_to_payload_with_future)

    return submit_tree_map(client, export_payload_tree, _run_export_task, pure=False)


def combine_results(
    current: Optional[GtsfmData], child_results: tuple[Optional[GtsfmData], ...]
) -> Optional[GtsfmData]:
    """Merge bundle adjustment outputs from child clusters into the parent result."""
    if len(child_results) == 0:
        return current

    logger.info("🫱🏻‍🫲🏽 Merging with %d children", len(child_results))
    for idx, child in enumerate(child_results):
        _log_scene_reprojection_stats(child, f"child #{idx}")
    merged = current
    for child in child_results:
        if child is None:
            continue
        if merged is None:
            merged = child
            continue
        try:
            # Use cameras to estimate a similarity transform between merged and child.
            aSb = align_utils.sim3_from_Pose3_maps(merged.poses(), child.poses())
        except Exception as exc:
            logger.warning("⚠️ Failed to align cluster outputs: %s", exc)
            aSb = Similarity3()  # identity
        try:
            merged = merged.merged_with(child, aSb)  # Should always succeed
        except Exception as exc:
            logger.warning("⚠️ Failed to merge cluster outputs: %s", exc)

    _log_scene_reprojection_stats(merged, "merged result (camera only)")
    return merged
    # # Done merging, now run BA to refine.
    # if merged is None:
    #     return None
    # try:
    #     merged_with_ba = BundleAdjustmentOptimizer().run_simple_ba(merged)[0]  # Can definitely fail
    #     _log_scene_reprojection_stats(merged_with_ba, "merged result (with ba)")
    #     return merged_with_ba
    # except Exception as e:
    #     logger.warning("⚠️ Failed to run bundle adjustment: %s", e)
    #     return merged


__all__ = ["combine_results", "schedule_exports"]
