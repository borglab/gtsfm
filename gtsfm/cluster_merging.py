"""Utilities for merging cluster reconstructions and exporting results."""

from __future__ import annotations

import re
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

_SCENE_PLOTS_DIR_ATTR = "_gtsfm_plots_dir"
_SCENE_LABEL_ATTR = "_gtsfm_cluster_label"


def _sanitize_component(value: str, fallback: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return sanitized or fallback


def annotate_scene_with_metadata(
    scene: Optional[GtsfmData],
    plots_dir: str | Path | None,
    cluster_label: Optional[str],
) -> Optional[GtsfmData]:
    """Attach plotting metadata to a scene before it is merged downstream."""
    if scene is None:
        return None
    if plots_dir is not None:
        setattr(scene, _SCENE_PLOTS_DIR_ATTR, Path(plots_dir))
    if cluster_label is not None:
        setattr(scene, _SCENE_LABEL_ATTR, cluster_label)
    return scene


def _propagate_scene_metadata(target: Optional[GtsfmData], *sources: Optional[GtsfmData]) -> None:
    """Ensure merged scenes keep the plots directory and label metadata."""
    if target is None:
        return
    for source in sources:
        if source is None:
            continue
        plots_dir = getattr(source, _SCENE_PLOTS_DIR_ATTR, None)
        label = getattr(source, _SCENE_LABEL_ATTR, None)
        if plots_dir is not None or label is not None:
            annotate_scene_with_metadata(target, plots_dir, label)
        if getattr(target, _SCENE_PLOTS_DIR_ATTR, None) is not None and getattr(
            target, _SCENE_LABEL_ATTR, None
        ) is not None:
            break


def _compute_scene_reprojection_stats(
    scene: Optional[GtsfmData],
) -> Optional[tuple[np.ndarray, float, float, float, float]]:
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
        all_errors,
        float(np.mean(all_errors)),
        float(np.median(all_errors)),
        float(np.min(all_errors)),
        float(np.max(all_errors)),
    )


def _log_scene_reprojection_stats(scene: Optional[GtsfmData], label: str, *, plot_histograms: bool) -> None:
    """Log reprojection error statistics."""
    stats = _compute_scene_reprojection_stats(scene)
    if stats is None:
        logger.info("📏 %s reprojection error stats unavailable", label)
        return
    errors, mean, median, min_err, max_err = stats
    logger.info(
        "📏 %s reprojection error (px): mean=%.2f median=%.2f min=%.2f max=%.2f",
        label,
        mean,
        median,
        min_err,
        max_err,
    )
    if plot_histograms:
        _plot_reprojection_error_distribution(errors, scene, label, mean, median, min_err, max_err)


def _plot_reprojection_error_distribution(
    errors: np.ndarray,
    scene: Optional[GtsfmData],
    label: str,
    mean: float,
    median: float,
    min_err: float,
    max_err: float,
) -> None:
    """Persist a histogram of reprojection errors for diagnostics."""
    if scene is None:
        return
    plots_dir: Optional[Path] = getattr(scene, _SCENE_PLOTS_DIR_ATTR, None)
    if plots_dir is None:
        return
    cluster_label = getattr(scene, _SCENE_LABEL_ATTR, None) or label
    sanitized_cluster = _sanitize_component(cluster_label, "scene")
    sanitized_context = _sanitize_component(label, "context")
    output_path = plots_dir / f"{sanitized_cluster}_{sanitized_context}_reprojection_error_hist.png"

    try:
        import matplotlib.pyplot as plt  # Local import, plotting only happens when needed

        plots_dir.mkdir(parents=True, exist_ok=True)
        bin_count = int(np.clip(np.sqrt(errors.size) * 2, 10, 120))
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.hist(errors, bins=bin_count, color="#1f77b4", edgecolor="white")
        ax.set_title(f"{cluster_label} reprojection error distribution")
        ax.set_xlabel("Reprojection error (px)")
        ax.set_ylabel("Track count")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        stats_text = "\n".join(
            [
                f"mean: {mean:.2f} px",
                f"median: {median:.2f} px",
                f"min: {min_err:.2f} px",
                f"max: {max_err:.2f} px",
            ]
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        logger.warning("⚠️ Failed to save reprojection error plot %s: %s", output_path, exc)


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
    current: Optional[GtsfmData],
    child_results: tuple[Optional[GtsfmData], ...],
    *,
    run_bundle_adjustment_on_parent: bool = True,
    plot_reprojection_histograms: bool = True,
) -> Optional[GtsfmData]:
    """Merge bundle adjustment outputs from child clusters into the parent result."""
    if len(child_results) == 0:
        return current

    logger.info("🫱🏻‍🫲🏽 Merging with %d children", len(child_results))
    for idx, child in enumerate(child_results):
        _log_scene_reprojection_stats(child, f"child #{idx}", plot_histograms=plot_reprojection_histograms)
    metadata_source = current if current is not None else next((c for c in child_results if c is not None), None)
    merged = current
    for child in child_results:
        if child is None:
            continue
        if child.number_tracks() == 0:
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

    _propagate_scene_metadata(merged, metadata_source)
    _log_scene_reprojection_stats(merged, "merged result (camera only)", plot_histograms=plot_reprojection_histograms)
    if merged is not None and merged.number_tracks() > 0:
        track_errors: list[float] = []
        tracks = merged.tracks()
        cameras = merged.cameras()
        for track in tracks:
            errors, _ = compute_track_reprojection_errors(cameras, track)
            mean_error = float(np.nanmean(errors)) if errors.size > 0 else np.nan
            track_errors.append(mean_error)
        finite_errors = np.array(track_errors, dtype=float)
        finite_errors = finite_errors[np.isfinite(finite_errors)]
        if finite_errors.size > 0:
            median_error = float(np.median(finite_errors))
            error_threshold = min(5.0, median_error * 5.0)
            retained_tracks = [
                track for track, err in zip(tracks, track_errors) if np.isfinite(err) and err <= error_threshold
            ]
            removed_count = merged.number_tracks() - len(retained_tracks)
            if removed_count > 0:
                logger.info(
                    "🧹 Dropping %d tracks with reprojection error > %.2f px (median=%.2f).",
                    removed_count,
                    error_threshold,
                    median_error,
                )
                filtered = GtsfmData(
                    merged.number_images(),
                    cameras=cameras,
                    tracks=retained_tracks,
                    gaussian_splats=merged.get_gaussian_splats(),
                )
                for idx in range(merged.number_images()):
                    info = merged.get_image_info(idx)
                    if info.name is not None or info.shape is not None:
                        filtered.set_image_info(idx, name=info.name, shape=info.shape)
                _propagate_scene_metadata(filtered, merged)
                merged = filtered

    if not run_bundle_adjustment_on_parent:
        return merged

    if merged is None:
        return None

    # Log cameras that have no supporting track measurements before running BA.
    all_cameras = set(merged.get_valid_camera_indices())
    cameras_with_measurements: set[int] = set()
    for track_idx in range(merged.number_tracks()):
        track = merged.get_track(track_idx)
        for m_idx in range(track.numberMeasurements()):
            cam_idx, _ = track.measurement(m_idx)
            cameras_with_measurements.add(cam_idx)
    zero_track_cameras = sorted(all_cameras - cameras_with_measurements)
    if zero_track_cameras:
        logger.warning("📋 Cameras with zero tracks before parent BA: %s", zero_track_cameras)
        if cameras_with_measurements:
            merged = GtsfmData.from_selected_cameras(merged, sorted(cameras_with_measurements))
            logger.info(
                "Pruned %d zero-track cameras; %d cameras remain for parent BA.",
                len(zero_track_cameras),
                len(merged.get_valid_camera_indices()),
            )
        else:
            logger.warning("All cameras lack tracks; skipping parent BA.")
            return merged
    else:
        logger.info("✅ All cameras have at least one track before parent BA.")

    try:
        merged_with_ba = BundleAdjustmentOptimizer().run_simple_ba(merged)[0]  # Can definitely fail
        _propagate_scene_metadata(merged_with_ba, merged)
        _log_scene_reprojection_stats(
            merged_with_ba,
            "merged result (with ba)",
            plot_histograms=plot_reprojection_histograms,
        )
        return merged_with_ba
    except Exception as exc:
        logger.warning("⚠️ Failed to run bundle adjustment: %s", exc)
        return merged


__all__ = [
    "combine_results",
    "schedule_exports",
    "annotate_scene_with_metadata",
]
