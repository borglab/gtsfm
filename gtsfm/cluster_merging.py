"""Utilities for merging cluster reconstructions and exporting results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from dask.distributed import Client, Future
from gtsam import Similarity3

import gtsfm.utils.logger as logger_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.cluster_optimizer.cluster_anysplat import save_splats
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils import align as align_utils
from gtsfm.utils.splat import GaussiansProtocol, merge_gaussian_splats
from gtsfm.utils.transform import transform_gaussian_splats
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
    if merged_scene.has_gaussian_splats():
        gaussian_splats = merged_scene.get_gaussian_splats()
        if isinstance(gaussian_splats, GaussiansProtocol):
            try:
                save_splats(merged_scene, merged_dir)
            except Exception as exc:
                logger.warning("⚠️ Failed to export Gaussian splats: %s", exc)


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

    # Done merging, now run BA to refine.
    if merged is None:
        return None
    try:
        post_ba_result, _ = BundleAdjustmentOptimizer().run_simple_ba(merged)
        try:
            post_ba_result.set_gaussian_splats(None)
            postba_S_current = align_utils.sim3_from_Pose3_maps(post_ba_result.poses(), current.poses())  # type: ignore
            merged_gaussians = transform_gaussian_splats(current.get_gaussian_splats(), postba_S_current)  # type: ignore
            for child in child_results:
                if child is None:
                    continue
                try:
                    postba_S_child = align_utils.sim3_from_Pose3_maps(post_ba_result.poses(), child.poses())

                    if child.has_gaussian_splats():
                        transformed_other_gaussians = transform_gaussian_splats(
                            child.get_gaussian_splats(), postba_S_child  # type: ignore
                        )
                        if merged_gaussians is None:
                            merged_gaussians = transformed_other_gaussians
                        else:
                            merged_gaussians = merge_gaussian_splats(merged_gaussians, transformed_other_gaussians)

                except Exception as e:
                    logger.warning("⚠️ Failed to align and merge gaussians: %s", e)
            post_ba_result.set_gaussian_splats(merged_gaussians)
            return post_ba_result  # Can definitely fail

        except Exception as alignment_exc:
            logger.warning("⚠️ Failed to compute pre/post BA Sim(3): %s", alignment_exc)
            return merged

    except Exception as e:
        logger.warning("⚠️ Failed to run bundle adjustment: %s", e)
        return merged


__all__ = ["combine_results", "schedule_exports"]
