"""Utilities for merging cluster reconstructions and exporting results."""

from __future__ import annotations

import re
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import gtsam
import numpy as np
from dask.distributed import Client, Future
from gtsam import Similarity3, Pose3, UnaryMeasurementPose3, TrajectoryAlignerSim3

import gtsfm.utils.logger as logger_utils
import gtsfm.common.types as gtsfm_types
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer, RobustBAMode
from gtsfm.cluster_optimizer.cluster_anysplat import save_splats
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.utils import align as align_utils
from gtsfm.utils import data_utils
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


def _create_unary_measurements(scene: GtsfmData) -> list[UnaryMeasurementPose3]:
    # TODO(akshay-krishnan): investigate using a scene-dependent noise model
    # perhaps * np.exp(-len(scene.get_valid_camera_indices()) / 100.0)

    unary_measurements = []
    camera_reproj_errors = scene.get_scene_reprojection_errors_per_camera()
    num_good_measurements = {
        cam_id: (~np.isnan(errors) & (errors < 2.0)).sum() for cam_id, errors in camera_reproj_errors.items()
    }
    for i, camera in scene.get_camera_poses().items():
        if camera is None:
            continue
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1]) / np.sqrt(num_good_measurements.get(i, 0) + 1e-6)
        )
        unary_measurement = UnaryMeasurementPose3(i, camera, noise_model)
        unary_measurements.append(unary_measurement)
    return unary_measurements


def merge_scenes_with_sim3_nonlinear(parent_scene: GtsfmData, children_scenes: list[GtsfmData]) -> GtsfmData:
    if len(children_scenes) == 0:
        return parent_scene

    aTi_measurements = _create_unary_measurements(parent_scene)
    parent_camera_ids = set(parent_scene.get_valid_camera_indices())
    valid_child_scenes = []

    for i, child_scene in enumerate(children_scenes):
        child_camera_ids = set(child_scene.get_valid_camera_indices())
        common_camera_ids = parent_camera_ids & child_camera_ids
        if len(common_camera_ids) == 0:
            logger.warning("Child scene %d has insufficient overlap with parent, skipping", i)
            continue
        valid_child_scenes.append(child_scene)

    if len(valid_child_scenes) == 0:
        return parent_scene

    aTi_measurements = _create_unary_measurements(parent_scene)
    bTi_measurements = [_create_unary_measurements(child_scene) for child_scene in valid_child_scenes]
    aligner = TrajectoryAlignerSim3(aTi_measurements, bTi_measurements)
    result = aligner.solve()

    opt_aTi = {i: result.atPose3(i) for i in parent_scene.get_valid_camera_indices() if i in result.keys()}

    merged = parent_scene
    for i, aTi in opt_aTi.items():
        merged.update_camera_pose(i, aTi)

    for i, child_scene in enumerate(valid_child_scenes):
        opt_bSa = result.atSimilarity3(gtsam.Symbol("S", i).key())
        opt_aSb = opt_bSa.inverse()
        merged = merged.merged_with(child_scene, opt_aSb)  # type: ignore
    return merged


@dataclass(frozen=True)
class MergedNodeResult:
    """Results of merging child scenes with parent scenes in the reconstruction tree.

    Attributes:
        scene: The merged scene.
        metrics: The metrics for the merged scene.
    """

    scene: Optional[GtsfmData]
    pre_ba_scene: Optional[GtsfmData]
    metrics: GtsfmMetricsGroup
    pre_ba_metrics: GtsfmMetricsGroup


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
    # Compute metrics for only those images that belong to the merged scene.
    image_idxs = list(result_data._image_info.keys())
    poses_gt: dict[int, Pose3] = {
        i: cameras_gt[i].pose() for i in image_idxs if i < len(cameras_gt) and cameras_gt[i] is not None
    }

    if len(poses_gt) == 0:
        return GtsfmMetricsGroup(name="ba_pose_error_metrics", metrics=[])

    aligned_result_data = result_data.align_via_sim3_and_transform(poses_gt)
    return metrics_utils.compute_ba_pose_metrics(
        gt_wTi=poses_gt, computed_wTi=aligned_result_data.get_camera_poses(), save_dir=save_dir, store_full_data=True
    )


def compute_merging_metrics(
    merged_scene: Optional[GtsfmData],
    *,
    cameras_gt: Optional[list[Optional[gtsfm_types.CAMERA_TYPE]]] = None,
    save_dir: Optional[str | Path] = None,
    store_full_data: bool = False,
    child_camera_counts: list[int] | None = None,
    child_camera_overlap_with_parent: list[int] | None = None,
    suffix: str = "",
) -> GtsfmMetricsGroup:
    """Build metrics describing a merged reconstruction at a tree node.

    Args:
        merged_scene: The merged scene.
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
    merging_metrics = GtsfmMetricsGroup(name=f"merging_metrics{suffix}", metrics=metrics)
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
    pre_ba_merged = merged_result.pre_ba_scene
    if pre_ba_merged is not None:
        pre_ba_dir = merged_dir.parent / "merged_pre_ba"
        pre_ba_dir.mkdir(parents=True, exist_ok=True)
        pre_ba_merged.export_as_colmap_text(pre_ba_dir)
        if pre_ba_merged.has_gaussian_splats():
            gaussian_splats = pre_ba_merged.get_gaussian_splats()
            if isinstance(gaussian_splats, GaussiansProtocol):
                try:
                    save_splats(pre_ba_merged, pre_ba_dir)
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


def _drop_outlier_tracks(scene: GtsfmData) -> GtsfmData:
    """Drop outlier tracks from a scene based on reprojection error.

    Args:
        scene: The scene to drop outlier tracks from.

    Returns:
        The scene with outlier tracks dropped.
    """
    if scene.number_tracks() == 0:
        return scene
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
            filtered = GtsfmData.from_cameras_and_tracks(
                cameras,
                retained_tracks,
                number_images=scene.number_images(),
                image_info=scene._clone_image_info(),
                gaussian_splats=scene.get_gaussian_splats(),
            )
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
    current: GtsfmData | None,
    child_results: tuple[MergedNodeResult, ...],
    *,
    cameras_gt: Optional[list[Optional[gtsfm_types.CAMERA_TYPE]]] = None,
    run_bundle_adjustment_on_parent: bool = True,
    plot_reprojection_histograms: bool = True,
    merge_duplicate_tracks: bool = True,
    drop_outlier_after_camera_merging: bool = True,
    drop_camera_with_no_track: bool = True,
    drop_child_if_merging_fail: bool = True,
    store_full_data: bool = False,
    use_nonlinear_sim3_alignment: bool = False,
    use_shared_calibration: bool = True,
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

    child_scenes: tuple[Optional[GtsfmData], ...] = tuple[GtsfmData | None, ...](child.scene for child in child_results)

    # Some stats for the merging metrics.
    parent_camera_set = set(current.get_valid_camera_indices()) if current is not None else set()
    child_camera_counts: list[int] = []
    child_camera_overlap_with_parent: list[int] = []
    for child_scene in child_scenes:
        child_cam_set = set(child_scene.get_valid_camera_indices()) if child_scene is not None else set()
        child_camera_counts.append(len(child_cam_set))
        child_camera_overlap_with_parent.append(len(child_cam_set & parent_camera_set))

    def _finalize_result(result_scene: Optional[GtsfmData], pre_ba_scene: Optional[GtsfmData]) -> MergedNodeResult:
        return MergedNodeResult(
            scene=result_scene,
            pre_ba_scene=pre_ba_scene,
            metrics=compute_merging_metrics(
                result_scene,
                cameras_gt=cameras_gt,
                store_full_data=store_full_data,
                child_camera_counts=child_camera_counts,
                child_camera_overlap_with_parent=child_camera_overlap_with_parent,
            ),
            pre_ba_metrics=compute_merging_metrics(
                pre_ba_scene,
                cameras_gt=cameras_gt,
                store_full_data=store_full_data,
                child_camera_counts=child_camera_counts,
                child_camera_overlap_with_parent=child_camera_overlap_with_parent,
                suffix="_pre_ba",
            ),
        )

    if current is None:
        return _finalize_result(None, None)

    # Log reprojection stats for the current scene and all children.
    _log_scene_reprojection_stats(current, "Current Node", plot_histograms=plot_reprojection_histograms)
    valid_child_scenes = [c for c in child_scenes if c is not None]

    logger.info("🫱🏻‍🫲🏽 Merging with %d / %d valid children ", len(valid_child_scenes), len(child_scenes))

    for idx, child in enumerate(child_scenes):
        if child is not None:
            _log_scene_reprojection_stats(child, f"child #{idx}", plot_histograms=plot_reprojection_histograms)

    if len(valid_child_scenes) == 0:
        return _finalize_result(current, None)

    metadata_source = current

    # Initialize the merged scene: use the current scene.
    merged = current
    _log_scene_reprojection_stats(merged, "Current node", plot_histograms=plot_reprojection_histograms)

    if use_nonlinear_sim3_alignment:
        merged = merge_scenes_with_sim3_nonlinear(merged, valid_child_scenes)
        _log_scene_reprojection_stats(
            merged, "Merged with children (nonlinear alignment)", plot_histograms=plot_reprojection_histograms
        )
    else:
        # Merge all children into the merged scene.
        for i, child in enumerate(valid_child_scenes):
            merged = _align_and_merge_results(merged, child, drop_if_merging_fails=drop_child_if_merging_fail)
            _log_scene_reprojection_stats(
                merged, f"Merged with child #{i+1}", plot_histograms=plot_reprojection_histograms
            )

    _propagate_scene_metadata(merged, metadata_source)

    if merged is None:
        return _finalize_result(None, None)

    if merge_duplicate_tracks and merged is not None and merged.number_tracks() > 0:
        original_track_count = merged.number_tracks()
        merged_tracks: list = []
        measurement_to_track: dict[tuple[int, int, int], int] = {}

        def _measurement_key(cam_idx: int, uv: np.ndarray) -> tuple[int, int, int]:
            return cam_idx, math.floor(float(uv[0])), math.floor(float(uv[1]))

        for track in merged.tracks():
            measurements = [track.measurement(k) for k in range(track.numberMeasurements())]
            target_idx = None
            for cam_idx, uv in measurements:
                existing_idx = measurement_to_track.get(_measurement_key(cam_idx, uv))
                if existing_idx is not None:
                    target_idx = existing_idx
                    break

            if target_idx is None:
                merged_tracks.append(track)
                target_idx = len(merged_tracks) - 1
            else:
                base_track = merged_tracks[target_idx]
                existing_cams = {base_track.measurement(m_idx)[0] for m_idx in range(base_track.numberMeasurements())}
                for cam_idx, uv in measurements:
                    if cam_idx in existing_cams:
                        continue
                    base_track.addMeasurement(cam_idx, uv)
                    existing_cams.add(cam_idx)

            base_track = merged_tracks[target_idx]
            for m_idx in range(base_track.numberMeasurements()):
                cam_idx, uv = base_track.measurement(m_idx)
                measurement_to_track[_measurement_key(cam_idx, uv)] = target_idx

        if len(merged_tracks) < original_track_count:
            logger.info(
                "🪢 Merged %d duplicate tracks into %d unique tracks after camera alignment.",
                original_track_count,
                len(merged_tracks),
            )
        valid_tracks: list = []
        for track in merged_tracks:
            if track.hasUniqueCameras():
                valid_tracks.append(track)
        if len(valid_tracks) < len(merged_tracks):
            logger.info(
                "🧹 Discarding %d invalid tracks with repeated camera observations.",
                len(merged_tracks) - len(valid_tracks),
            )
        merged._tracks = valid_tracks

    if not run_bundle_adjustment_on_parent:
        if drop_outlier_after_camera_merging:
            merged = _drop_outlier_tracks(merged)
        return _finalize_result(merged, None)

    # Log cameras that have no supporting track measurements before running BA.
    if drop_camera_with_no_track:
        merged, should_run_ba = data_utils.remove_cameras_with_no_tracks(merged, "parent BA")
        if not should_run_ba:
            return _finalize_result(merged, None)
    else:
        logger.info("📌 Retaining zero-track cameras before parent BA (drop disabled).")

    try:
        optimizer = BundleAdjustmentOptimizer(
            robust_ba_mode=RobustBAMode.HUBER,
            calibration_prior_focal_sigma=10.0,
            use_calibration_prior=True,
            shared_calib=use_shared_calibration,
            robust_noise_basin=0.5,
        )
        merged_with_ba, _ = optimizer.run_simple_ba(merged)
        _propagate_scene_metadata(merged_with_ba, merged)
        _log_scene_reprojection_stats(
            merged_with_ba,
            "merged result (with ba)",
            plot_histograms=plot_reprojection_histograms,
        )
        if drop_outlier_after_camera_merging:
            merged_with_ba = _drop_outlier_tracks(merged_with_ba)
            _log_scene_reprojection_stats(
                merged_with_ba,
                "merged result (with ba + outlier filtering)",
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
                return _finalize_result(merged_with_ba, merged)

            except Exception as alignment_exc:
                logger.warning("⚠️ Failed to compute pre/post BA Sim(3): %s", alignment_exc)
                return _finalize_result(merged_with_ba, merged)

        else:
            logger.info("✖️ No Gaussians to merge")
            return _finalize_result(merged_with_ba, merged)
    except Exception as exc:
        logger.warning("⚠️ Failed to run bundle adjustment: %s", exc)
        return _finalize_result(merged, None)


__all__ = [
    "MergedNodeResult",
    "combine_results",
    "schedule_exports",
    "compute_merging_metrics",
    "annotate_scene_with_metadata",
]
