"""Utilities for merging cluster reconstructions and exporting results."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from dask.distributed import Client, Future
from gtsam import Similarity3

import gtsfm.utils.logger as logger_utils
import gtsfm.common.types as gtsfm_types
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.cluster_optimizer.cluster_anysplat import save_splats
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.utils import align as align_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.utils.reprojection import compute_track_reprojection_errors
from gtsfm.utils.splat import GaussiansProtocol, merge_gaussian_splats
from gtsfm.utils.transform import transform_gaussian_splats
from gtsfm.utils.tree import Tree
from gtsfm.utils.tree_dask import submit_tree_map

if TYPE_CHECKING:
    from gtsfm.scene_optimizer import ClusterExecutionHandles

logger = logger_utils.get_logger()

_SCENE_PLOTS_DIR_ATTR = "_gtsfm_plots_dir"
_SCENE_LABEL_ATTR = "_gtsfm_cluster_label"


@dataclass(frozen=True)
class MergedNodeResult:
    """Results of merging child scenes with parent scenes in the reconstruction tree.
    
    Attributes:
        scene: The merged scene.
        metrics: The metrics for the merged scene.
    """

    scene: Optional[GtsfmData]
    metrics: GtsfmMetricsGroup


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
        if (
            getattr(target, _SCENE_PLOTS_DIR_ATTR, None) is not None
            and getattr(target, _SCENE_LABEL_ATTR, None) is not None
        ):
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
    assert scene is not None
    logger.info(
        "📏 %s reprojection error (px): mean=%.2f median=%.2f min=%.2f max=%.2f, #cameras=%d, #tracks=%d, #images=%d",
        label,
        mean,
        median,
        min_err,
        max_err,
        len(scene.cameras()),
        scene.number_tracks(),
        scene.number_images(),
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


def _get_pose_metrics(
    result_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    save_dir: Optional[str] = None,
) -> GtsfmMetricsGroup:
    """Compute pose metrics for a BA result after aligning with ground truth.
    
    Args:
        result_data: The GtsfmData object to compute pose metrics for.
        cameras_gt: The ground truth cameras.
        save_dir: The directory to save the pose metrics to.

    Returns:
        A GtsfmMetricsGroup object containing the pose metrics.
    """
    # TODO: this uses all cameras in the ground truth, but we only need the ones that are present in input to this node.
    # TODO: this can only be fixed if we use a field in result data to store the valid input images.
    poses_gt = [cam.pose() if cam is not None else None for cam in cameras_gt]

    if len(poses_gt) == 0:
        return GtsfmMetricsGroup(name="ba_pose_error_metrics", metrics=[])

    aligned_result_data = result_data.align_via_sim3_and_transform(poses_gt)
    return metrics_utils.compute_ba_pose_metrics(
        gt_wTi_list=poses_gt,
        computed_wTi=aligned_result_data.get_camera_poses(),
        save_dir=save_dir,
    )


def compute_merging_metrics(
    merged_scene: Optional[GtsfmData],
    *,
    cameras_gt: Optional[list[Optional[gtsfm_types.CAMERA_TYPE]]] = None,
    save_dir: Optional[str | Path] = None,
    store_full_data: bool = False,
    child_camera_counts: list[int] | None = None,
    child_camera_overlap_with_parent: list[int] | None = None,
) -> GtsfmMetricsGroup:
    """Build metrics describing a merged reconstruction at a tree node.
    
    Args:
        merged_scene: The merged scene.
        child_count: The number of children.
        cameras_gt: The ground truth cameras.
        save_dir: The directory to save the metrics to.
        store_full_data: Whether to store full data.
        child_camera_counts: The number of cameras in each child.
        child_camera_overlap_with_parent: The number of cameras in the parent that are also in each child.

    Returns:
        A GtsfmMetricsGroup object containing the merging metrics.
    """
    child_camera_counts = child_camera_counts or []
    child_camera_overlap_with_parent = child_camera_overlap_with_parent or []
    child_camera_counts_arr = np.asarray(child_camera_counts, dtype=np.int32)
    child_camera_overlap_arr = np.asarray(child_camera_overlap_with_parent, dtype=np.int32)
    pose_save_dir = str(save_dir) if isinstance(save_dir, Path) else save_dir
    metrics = [
        GtsfmMetric("merge_success", 1 if merged_scene is not None else 0),
        GtsfmMetric("merge_child_count", len(child_camera_counts)),
        GtsfmMetric(
            "child_camera_counts",
            child_camera_counts_arr,
            store_full_data=True,
        ),
        GtsfmMetric(
            "child_camera_overlap_with_parent",
            child_camera_overlap_arr,
            store_full_data=True,
        ),
    ]
    if merged_scene is not None:
        metrics.extend(merged_scene.get_metrics(suffix="_merged", store_full_data=store_full_data))
    merging_metrics = GtsfmMetricsGroup(name="merging_metrics", metrics=metrics)
    if cameras_gt is not None and merged_scene is not None:
        ba_pose_error_metrics = _get_pose_metrics(
            merged_scene,
            cameras_gt,
            save_dir=pose_save_dir,
        )
        merging_metrics.extend(ba_pose_error_metrics)
    return merging_metrics


def _run_export_task(payload: Tuple[Optional[Path], Future | MergedNodeResult]) -> None:
    """Persist a merged reconstruction to COLMAP text format.

    Args:
        payload: Tuple pairing the directory, and the future or result of the merged reconstruction.
    """
    merged_dir, merged_future = payload
    merged_result = merged_future.result() if isinstance(merged_future, Future) else merged_future
    merged_scene = merged_result.scene
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


def _remove_cameras_with_no_tracks(scene: GtsfmData) -> tuple[GtsfmData, bool]:
    """Remove cameras with no tracks from a scene.

    Args:
        scene: The scene to remove cameras from.

    Returns:
        A tuple containing the scene with cameras removed and a boolean indicating if the scene should run BA.
    """
    all_cameras = set(scene.get_valid_camera_indices())
    camera_measurement_map = scene.get_camera_to_measurement_map()
    cameras_with_measurements = set(camera_measurement_map.keys())
    zero_track_cameras = sorted(all_cameras - cameras_with_measurements)
    if zero_track_cameras:
        logger.warning("📋 Cameras with zero tracks before parent BA: %s", zero_track_cameras)
    if zero_track_cameras:
        if cameras_with_measurements:
            scene = GtsfmData.from_selected_cameras(scene, sorted(cameras_with_measurements))
            logger.info(
                "Pruned %d zero-track cameras; %d cameras remain for parent BA.",
                len(zero_track_cameras),
                len(scene.get_valid_camera_indices()),
            )
        else:
            logger.warning("All cameras lack tracks; skipping parent BA.")
            return scene, False
    elif not zero_track_cameras:
        logger.info("✅ All cameras have at least one track before parent BA.")

    return scene, True


def _drop_outlier_tracks(scene: GtsfmData) -> GtsfmData:
    """Drop outlier tracks from a scene based on reprojection error.

    Args:
        scene: The scene to drop outlier tracks from.

    Returns:
        The scene with outlier tracks dropped.
    """
    track_errors: list[float] = []
    tracks = scene.tracks()
    cameras = scene.cameras()
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
        removed_count = scene.number_tracks() - len(retained_tracks)
        if removed_count > 0:
            logger.info(
                "🧹 Dropping %d tracks with reprojection error > %.2f px (median=%.2f).",
                removed_count,
                error_threshold,
                median_error,
            )
            filtered = GtsfmData(
                scene.number_images(),
                cameras=cameras,
                tracks=retained_tracks,
                gaussian_splats=scene.get_gaussian_splats(),
            )
            for idx in scene.get_valid_camera_indices():
                info = scene.get_image_info(idx)
                if info.name is not None or info.shape is not None:
                    filtered.set_image_info(idx, name=info.name, shape=info.shape)
            _propagate_scene_metadata(filtered, scene)
            scene = filtered
    return scene


def _align_and_merge_results(result1: GtsfmData, result2: GtsfmData, drop_if_merging_fails: bool = True) -> GtsfmData:
    """Align result2 to result1 and merge it with result1.

    Args:
        result1: The first result to merge.
        result2: The second result to merge.
        drop_if_merging_fails: Whether to drop result2 if merging fails.
    
    Returns:
        The merged result if merging succeeds, otherwise the first result.
    """
    try:
        _1S2 = align_utils.sim3_from_Pose3_maps(result1.poses(), result2.poses())
    except Exception as exc:
        if drop_if_merging_fails:
            logger.warning("⚠️ Dropping a node because the two results are not aligned: %s", exc)
            return result1
        else:
            _1S2 = Similarity3()
    try:
        merged = result1.merged_with(result2, _1S2)
        return merged
    except Exception as exc:
        logger.warning("⚠️ Failed to merge results: %s", exc)
        return result1


def combine_results(
    current: Optional[GtsfmData],
    child_results: tuple[MergedNodeResult, ...],
    *,
    cameras_gt: Optional[list[Optional[gtsfm_types.CAMERA_TYPE]]] = None,
    run_bundle_adjustment_on_parent: bool = True,
    plot_reprojection_histograms: bool = True,
    drop_outlier_after_camera_merging: bool = True,
    drop_camera_with_no_track: bool = True,
    drop_child_if_merging_fail: bool = True,
    store_full_data: bool = False,
) -> MergedNodeResult:
    """Run the merging and parent BA pipeline using already-transformed children.
    
    Args:
        current: The current scene.
        child_results: The results of the child clusters.
        cameras_gt: The ground truth cameras.
        run_bundle_adjustment_on_parent: Whether to run bundle adjustment on the parent.
        plot_reprojection_histograms: Whether to plot the reprojection error histograms.
        drop_outlier_after_camera_merging: Whether to drop outlier tracks after camera merging.
        drop_camera_with_no_track: Whether to drop cameras with no tracks.
        drop_child_if_merging_fail: Whether to drop child scenes if merging fails.
        store_full_data: Whether to store full data for the merging metrics.

    Returns:
        A MergedNodeResult object containing the merged scene and its metrics.
    """

    child_scenes: tuple[Optional[GtsfmData], ...] = tuple(child.scene for child in child_results)

    # Some stats for the merging metrics.
    parent_camera_set = set(current.get_valid_camera_indices()) if current is not None else set()
    child_camera_counts: list[int] = []
    child_camera_overlap_with_parent: list[int] = []
    for child_scene in child_scenes:
        child_cam_set = set(child_scene.get_valid_camera_indices()) if child_scene is not None else set()
        child_camera_counts.append(len(child_cam_set))
        child_camera_overlap_with_parent.append(len(child_cam_set & parent_camera_set))

    def _finalize_result(result_scene: Optional[GtsfmData]) -> MergedNodeResult:
        return MergedNodeResult(
            result_scene,
            compute_merging_metrics(
                result_scene,
                cameras_gt=cameras_gt,
                store_full_data=store_full_data,
                child_camera_counts=child_camera_counts,
                child_camera_overlap_with_parent=child_camera_overlap_with_parent,
            ),
        )

    # Log reprojection stats for the current scene and all children.
    _log_scene_reprojection_stats(current, "Current Node", plot_histograms=plot_reprojection_histograms)
    valid_child_scenes = [c for c in child_scenes if c is not None]

    logger.info("🫱🏻‍🫲🏽 Merging with %d / %d valid children ", len(valid_child_scenes), len(child_scenes))

    for idx, child in enumerate(child_scenes):
        if child is not None:
            _log_scene_reprojection_stats(child, f"child #{idx}", plot_histograms=plot_reprojection_histograms)

    if len(valid_child_scenes) == 0:
        return _finalize_result(current)

    metadata_source = current

    # Initialize the merged scene: pick the first child if present, otherwise use the current scene.
    if len(valid_child_scenes) == 0:
        merged = current
        merged_is_current_frame = True
    else:
        merged = valid_child_scenes[0]
        valid_child_scenes.pop(0)
        merged_is_current_frame = False

    # Merge all children into the merged scene.
    for child in valid_child_scenes:
        assert merged is not None
        merged = _align_and_merge_results(merged, child, drop_if_merging_fails=drop_child_if_merging_fail)
    
    # If merged did not start in the current frame, merge it with the current scene.
    if not merged_is_current_frame and current is not None:
        assert merged is not None
        merged = _align_and_merge_results(merged, current, drop_if_merging_fails=drop_child_if_merging_fail)

    _propagate_scene_metadata(merged, metadata_source)
    _log_scene_reprojection_stats(merged, "merged result (camera only)", plot_histograms=plot_reprojection_histograms)

    if drop_outlier_after_camera_merging and merged is not None and merged.number_tracks() > 0:
        merged = _drop_outlier_tracks(merged)

    if not run_bundle_adjustment_on_parent:
        return _finalize_result(merged)

    if merged is None:
        return _finalize_result(None)

    # Log cameras that have no supporting track measurements before running BA.
    if drop_camera_with_no_track:
        merged, should_run_ba = _remove_cameras_with_no_tracks(merged)
        if not should_run_ba:
            return _finalize_result(merged)
    else:
        logger.info("📌 Retaining zero-track cameras before parent BA (drop disabled).")

    try:
        merged_with_ba = BundleAdjustmentOptimizer().run_simple_ba(merged)[0]  # Can definitely fail
        for idx in merged.get_valid_camera_indices():
            info = merged.get_image_info(idx)
            merged_with_ba.set_image_info(idx, name=info.name, shape=info.shape)
        _propagate_scene_metadata(merged_with_ba, merged)
        _log_scene_reprojection_stats(
            merged_with_ba,
            "merged result (with ba)",
            plot_histograms=plot_reprojection_histograms,
        )
        # TODO: the order here is different from the merging order above, we should fix this.
        if merged.has_gaussian_splats():
            logger.info("🫱🏻‍🫲🏽 Merging Gaussians")
            try:
                merged_with_ba.set_gaussian_splats(None)
                merged_gaussians = None
                if current is not None and current.has_gaussian_splats():
                    postba_S_current = align_utils.sim3_from_Pose3_maps(merged_with_ba.poses(), current.poses())
                    current_gaussians = current.get_gaussian_splats()
                    if current_gaussians is not None:
                        merged_gaussians = transform_gaussian_splats(current_gaussians, postba_S_current)

                for child in valid_child_scenes:
                    try:
                        postba_S_child = align_utils.sim3_from_Pose3_maps(merged_with_ba.poses(), child.poses())

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
                merged_with_ba.set_gaussian_splats(merged_gaussians)
                return _finalize_result(merged_with_ba)

            except Exception as alignment_exc:
                logger.warning("⚠️ Failed to compute pre/post BA Sim(3): %s", alignment_exc)
                return _finalize_result(merged)

        else:
            logger.info("✖️ No Gaussians to merge")
            return _finalize_result(merged_with_ba)
    except Exception as exc:
        logger.warning("⚠️ Failed to run bundle adjustment: %s", exc)
        return _finalize_result(merged)


__all__ = [
    "MergedNodeResult",
    "combine_results",
    "schedule_exports",
    "compute_merging_metrics",
    "annotate_scene_with_metadata",
]
