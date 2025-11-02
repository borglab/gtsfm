"""Utilities for merging cluster reconstructions and exporting results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from dask.distributed import Client, Future, get_client

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.utils import align as align_utils
from gtsfm.utils.tree import Tree
from gtsfm.utils.tree_dask import submit_tree_map

if TYPE_CHECKING:
    from gtsfm.scene_optimizer import ClusterExecutionHandles

logger = logger_utils.get_logger()


def _export_scene(
    merged_scene: Optional[GtsfmData],
    target_dir: Path,
    images: Optional[Sequence[Image]] = None,
) -> None:
    """Persist a merged reconstruction to COLMAP text format."""
    if merged_scene is None:
        return

    merged_path = Path(target_dir)
    merged_path.mkdir(parents=True, exist_ok=True)
    merged_scene.export_as_colmap_text(merged_path)


def _run_export_task(
    payload: tuple[ClusterExecutionHandles, Optional[GtsfmData], Sequence[Image] | Sequence[Future] | None],
) -> None:
    """Execute merged scene export on a worker."""
    handle, merged_scene, images = payload
    resolved_images: Sequence[Image] | None
    if images is None:
        resolved_images = None
    elif any(isinstance(img, Future) for img in images):
        try:
            resolved_images = tuple(get_client().gather(list(images)))
        except Exception:
            logger.warning("Failed to gather images for export; falling back to stored track colors.")
            resolved_images = None
    else:
        resolved_images = images
    merged_dir = handle.output_paths.results / "merged"
    _export_scene(merged_scene, merged_dir, resolved_images)


def schedule_exports(
    *,
    client: Client,
    handles_tree: Tree[ClusterExecutionHandles],
    merged_tree: Tree[Future],
    image_future_map: Mapping[int, Future],
) -> Tree[Future]:
    """Schedule persistence of merged reconstructions for each cluster."""
    export_payload_tree = Tree.zip(handles_tree, merged_tree).map(
        lambda value: (
            value[0],
            value[1],
            tuple(image_future_map[idx] for idx in sorted(image_future_map.keys())),
        )
    )

    return submit_tree_map(client, export_payload_tree, _run_export_task, pure=False)


def combine_results(
    current: Optional[GtsfmData], child_results: tuple[Optional[GtsfmData], ...]
) -> Optional[GtsfmData]:
    """Merge bundle adjustment outputs from child clusters into the parent result."""
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
    return merged


__all__ = ["combine_results", "schedule_exports"]
