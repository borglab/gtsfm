"""Utilities for merging cluster reconstructions and exporting results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from dask.distributed import Client, Future

import gtsfm.utils.logger as logger_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils import align as align_utils
from gtsfm.utils.tree import Tree
from gtsfm.utils.tree_dask import submit_tree_map

if TYPE_CHECKING:
    from gtsfm.scene_optimizer import ClusterExecutionHandles

logger = logger_utils.get_logger()


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

    merged = current
    for child in child_results:
        if child is None:
            continue
        if merged is None:
            merged = child
            continue
        try:
            aSb = align_utils.sim3_from_Pose3_maps(merged.poses(), child.poses())
            merged = merged.merged_with(child, aSb)
        except Exception as exc:
            logger.warning("Failed to merge cluster outputs: %s", exc)
    if merged is None:
        return None
    try:
        return BundleAdjustmentOptimizer().run_simple_ba(merged)[0]
    except Exception as e:
        logger.warning("Failed to run bundle adjustment: %s", e)
        return merged


__all__ = ["combine_results", "schedule_exports"]
