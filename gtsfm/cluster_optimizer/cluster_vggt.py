"""VGGT-based cluster optimizer leveraging the demo VGGT pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, Optional

import numpy as np
import torch
from dask.delayed import Delayed, delayed
from gtsam import Pose3
from PIL import Image as PILImage

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.metrics as metrics_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptions, multi_view_retriangulate_from_2d_tracks
from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterComputationGraph, ClusterContext, ClusterOptimizerBase
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.multi_view_tracker import MultiViewTracker
from gtsfm.frontend.vggt_geometry_transformer import (
    VggtGeometryTransformer,
    high_confidence_pointcloud,
    load_image_batch_vggt_loader,
    load_model,
    offload_vggt_model,
)
from gtsfm.products.visibility_graph import visibility_graph_keys
from gtsfm.ui.gtsfm_process import UiMetadata
from gtsfm.utils import data_utils
from gtsfm.utils.logger import get_logger

logger = get_logger()

# Module-level cache to avoid reloading VGGT weights per cluster.
_VGGT_MODEL_CACHE: dict[Hashable, Any] = {}


# ---------------------------------------------------------------------------
# Cluster-level BA runner
# ---------------------------------------------------------------------------


def _run_cluster_ba(
    gtsfm_data: GtsfmData,
    *,
    ba_options: BundleAdjustmentOptions,
    pre_ba_max_reproj_error: float = 0.0,
    post_ba_max_reproj_error: float = 3.0,
    drop_camera_with_no_track: bool = False,
    min_track_length: int = 2,
    cluster_label: Optional[str] = None,
    tracks_2d: Optional[list[SfmTrack2d]] = None,
    use_multi_view_retriangulation: bool = False,
) -> tuple[GtsfmData, GtsfmData]:
    """Run cluster-level BA on a GtsfmData result.

    This is a module-level function so it can be used with ``dask.delayed``.

    Args:
        tracks_2d: (optional) Union-find 2D tracks. Required when
            ``use_multi_view_retriangulation=True``.
        use_multi_view_retriangulation: When True, after the initial BA, re-triangulate
            ``tracks_2d`` against the post-BA cameras (recovers tracks dropped earlier
            in the pipeline) and run a second BA on the augmented track set.

    Returns:
        Tuple of (post_ba_result, pre_ba_result).
    """
    pre_ba_data = gtsfm_data

    if gtsfm_data.number_tracks() == 0:
        logger.warning("Skipping bundle adjustment because no valid tracks were produced.")
        return gtsfm_data, pre_ba_data

    if pre_ba_max_reproj_error > 0.0:
        num_tracks_before = gtsfm_data.number_tracks()
        gtsfm_data = gtsfm_data.filter_landmark_measurements(
            pre_ba_max_reproj_error, min_track_length
        )
        cluster_prefix = f"[{cluster_label}] " if cluster_label else ""
        logger.info(
            "%s🔍 #valid tracks after pre-BA reproj error filtering: %d out of %d",
            cluster_prefix,
            gtsfm_data.number_tracks(),
            num_tracks_before,
        )

    if drop_camera_with_no_track:
        gtsfm_data, should_run_ba = data_utils.remove_cameras_with_no_tracks(gtsfm_data, "cluster-level BA")
        if not should_run_ba:
            return gtsfm_data, pre_ba_data

    try:
        optimizer = ba_options.to_optimizer(min_track_length=min_track_length)
        gtsfm_data_with_ba, _ = optimizer.run_simple_ba(gtsfm_data)

        gtsfm_data_with_ba = gtsfm_data_with_ba.filter_landmark_measurements(
            post_ba_max_reproj_error
        )

        # Optional retri stage: re-triangulate union-find tracks against the post-BA
        # cameras and run another BA on the augmented set. Recovers tracks dropped
        # earlier in the pipeline; mirrors the retri stage in
        # BundleAdjustmentOptimizer._run_ba_and_evaluate.
        if use_multi_view_retriangulation and tracks_2d is not None:
            retri_data = multi_view_retriangulate_from_2d_tracks(
                gtsfm_data_with_ba, tracks_2d, min_track_length=min_track_length,
            )
            if retri_data.number_tracks() > 0:
                gtsfm_data_with_ba, _ = optimizer.run_simple_ba(retri_data)
                gtsfm_data_with_ba = gtsfm_data_with_ba.filter_landmark_measurements(
                    post_ba_max_reproj_error
                )

        logger.info(
            "%s🔍 #valid tracks after BA: %d out of %d",
            f"[{cluster_label}] " if cluster_label else "",
            gtsfm_data_with_ba.number_tracks(),
            gtsfm_data.number_tracks(),
        )
        return gtsfm_data_with_ba, pre_ba_data
    except Exception as exc:
        logger.warning("⚠️ Failed to run bundle adjustment: %s", exc)
        return gtsfm_data, pre_ba_data


# ---------------------------------------------------------------------------
# Dask helper functions
# ---------------------------------------------------------------------------


def _load_vggt_inputs(
    loader,
    indices: list[int],
    mode: str,
    *,
    save_processed_image: bool = False,
    output_root: Optional[str] = None,
    image_names: Optional[tuple[str, ...]] = None,
):
    """Load and preprocess a batch of images for VGGT."""
    image_batch, original_coords = load_image_batch_vggt_loader(loader, indices, mode=mode)
    if not save_processed_image or output_root is None or image_names is None:
        return image_batch, original_coords
    if len(image_names) != image_batch.shape[0]:
        logger.warning(
            "Skipping processed-image dump due to length mismatch: got %d names for %d images.",
            len(image_names),
            image_batch.shape[0],
        )
        return image_batch, original_coords

    target_root = Path(output_root) / "processed_images"
    target_root.mkdir(parents=True, exist_ok=True)
    batch_uint8 = (
        image_batch.detach().clamp(0.0, 1.0).mul(255.0).add(0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    )
    for i, image_name in enumerate(image_names):
        relpath = Path(image_name)
        save_path = target_root / relpath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(batch_uint8[i]).save(save_path)
    original_coords_np = original_coords.detach().cpu().numpy()
    num_coord_cols = original_coords_np.shape[1]
    if num_coord_cols == 6:
        coord_headers = ["left", "top", "right", "bottom", "scaled_width", "scaled_height"]
    else:
        coord_headers = [f"coord_{i}" for i in range(num_coord_cols)]
    rows = np.empty((len(image_names), num_coord_cols + 1), dtype=object)
    rows[:, 0] = np.asarray(image_names, dtype=object)
    rows[:, 1:] = original_coords_np
    np.savetxt(
        target_root / "original_coords.txt",
        rows,
        fmt=["%s"] + ["%.8f"] * num_coord_cols,
        delimiter="\t",
        header="\t".join(["image_name", *coord_headers]),
        comments="",
    )
    return image_batch, original_coords


def _resolve_vggt_model(cache_key: Hashable | None, loader_kwargs: dict[str, Any] | None) -> Any | None:
    """Fetch (or lazily load) a VGGT model for the current worker."""
    if cache_key is None:
        return None
    if cache_key in _VGGT_MODEL_CACHE:
        return _VGGT_MODEL_CACHE[cache_key]
    logger.info("⏳ Loading VGGT model weights...")
    loader_kwargs = loader_kwargs or {}
    model = load_model(**loader_kwargs)
    _VGGT_MODEL_CACHE[cache_key] = model
    logger.info("✅ VGGT model weights loaded successfully.")
    return model


def _run_vggt_pipeline(
    image_batch: torch.Tensor,
    original_coords: torch.Tensor,
    *,
    transformer: VggtGeometryTransformer,
    tracker: MultiViewTracker,
    image_indices: tuple[int, ...],
    image_names: tuple[str, ...] | None = None,
    dataset_dir: str | None = None,
    seed: int = 42,
    model_cache_key: Hashable | None = None,
    loader_kwargs: dict[str, Any] | None = None,
    weights_path: Any = None,
    cluster_label: Optional[str] = None,
) -> GtsfmData:
    """Run VGGT geometry prediction + tracking -> GtsfmData (no BA).

    This is a module-level function for use with ``dask.delayed``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cluster_label is not None:
        logger.info("🔵 Running VGGT on %s with %d images.", str(cluster_label).lower(), image_batch.shape[0])
    elif image_indices:
        logger.info("🔵 Running VGGT on %d images for partition %s.", image_batch.shape[0], image_indices)
    else:
        logger.info("🔵 Running VGGT on %d images.", image_batch.shape[0])

    cached_model = _resolve_vggt_model(model_cache_key, loader_kwargs)

    # Step 1: Geometry prediction.
    geo_output = transformer.predict(image_batch, model=cached_model, weights_path=weights_path)

    # Step 2: Optional tracking.
    tracking_result = None
    if tracker.config.tracking:
        tracking_result = tracker.run_tracking(
            geo_output,
            model=cached_model,
            image_names=image_names,
            dataset_dir=dataset_dir,
        )
        if geo_output.device.type == "cuda":
            offload_vggt_model(cached_model)

    # Step 3: Extract point cloud.
    points_3d, points_rgb = high_confidence_pointcloud(
        geo_output,
        confidence_threshold=transformer.config.confidence_threshold,
        max_num_points=transformer.config.max_num_points,
    )

    # Step 4: Assemble GtsfmData.
    gtsfm_data = tracker.build_gtsfm_data(
        geo_output,
        original_coords,
        image_indices=image_indices,
        image_names=image_names,
        tracking_result=tracking_result,
        points_3d=points_3d,
        points_rgb=points_rgb,
        cluster_label=cluster_label,
    )

    if geo_output.device.type == "cuda":
        del geo_output
        torch.cuda.empty_cache()

    return gtsfm_data


def _save_reconstruction_as_text(
    result: GtsfmData,
    results_path: Path,
    *,
    subdir: str = "vggt",
) -> None:
    target_dir = results_path / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    result.export_as_colmap_text(target_dir)


def _save_pre_ba_reconstruction_as_text(
    pre_ba_result: GtsfmData,
    results_path: Path,
) -> None:
    _save_reconstruction_as_text(pre_ba_result, results_path, subdir="vggt_pre_ba")


def _get_pose_metrics(
    result_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    save_dir: Optional[str] = None,
    metric_constructed_only: bool = False,
) -> GtsfmMetricsGroup:
    """Compute pose metrics for a VGGT result after aligning with ground truth."""
    image_idxs = list(result_data._image_info.keys())
    poses_gt: dict[int, Pose3] = {}
    for i in image_idxs:
        if i >= len(cameras_gt):
            continue
        camera = cameras_gt[i]
        if camera is not None:
            poses_gt[i] = camera.pose()
    if len(poses_gt) == 0:
        return GtsfmMetricsGroup(name="ba_pose_error_metrics", metrics=[])
    aligned_result_data = result_data.align_via_sim3_and_transform(poses_gt)
    computed_wTi: dict[int, Optional[Pose3]] = {i: pose for i, pose in aligned_result_data.get_camera_poses().items()}
    return metrics_utils.compute_ba_pose_metrics(
        gt_wTi=poses_gt,
        computed_wTi=computed_wTi,
        save_dir=save_dir,
        store_full_data=True,
        metric_constructed_only=metric_constructed_only,
    )


def _get_intrinsics_metrics(
    result_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
) -> GtsfmMetricsGroup:
    """Compute intrinsics metrics for a VGGT result against ground truth cameras."""
    image_idxs = list(result_data._image_info.keys())
    gt_cameras: dict[int, gtsfm_types.CAMERA_TYPE] = {}
    computed_cameras: dict[int, gtsfm_types.CAMERA_TYPE] = {}
    for i in image_idxs:
        if i >= len(cameras_gt):
            continue
        gt_cam = cameras_gt[i]
        est_cam = result_data.get_camera(i)
        if gt_cam is not None and est_cam is not None:
            gt_cameras[i] = gt_cam
            computed_cameras[i] = est_cam
    if len(gt_cameras) == 0:
        return GtsfmMetricsGroup(name="intrinsics_metrics", metrics=[])
    return metrics_utils.compute_intrinsics_metrics(
        gt_cameras=gt_cameras,
        computed_cameras=computed_cameras,
        store_full_data=True,
    )


def _aggregate_vggt_metrics(
    result: GtsfmData,
    cameras_gt: Optional[list[Optional[gtsfm_types.CAMERA_TYPE]]] = None,
    pre_ba_result: Optional[GtsfmData] = None,
    *,
    save_dir: Optional[str] = None,
    metric_constructed_only: bool = False,
) -> list[GtsfmMetricsGroup]:
    """Aggregate VGGT metrics into groups for both pre- and post-BA results."""
    def _build_metrics_group(scene: GtsfmData, name: str) -> GtsfmMetricsGroup:
        metrics_group = GtsfmMetricsGroup(
            name,
            [
                GtsfmMetric("num_cameras", len(scene.get_valid_camera_indices())),
                GtsfmMetric("num_points3d", scene.number_tracks()),
            ],
        )
        if cameras_gt is not None:
            metrics_group.extend(
                _get_pose_metrics(scene, cameras_gt, save_dir=save_dir, metric_constructed_only=metric_constructed_only)
            )
            metrics_group.extend(_get_intrinsics_metrics(scene, cameras_gt))
        return metrics_group

    metrics_groups = [_build_metrics_group(result, "cluster_vggt_metrics")]
    if pre_ba_result is not None:
        metrics_groups.append(_build_metrics_group(pre_ba_result, "cluster_vggt_pre_ba_metrics"))
    return metrics_groups


# ---------------------------------------------------------------------------
# ClusterVGGT
# ---------------------------------------------------------------------------


class ClusterVGGT(ClusterOptimizerBase):
    """Cluster optimizer that runs VGGT to generate COLMAP-style reconstructions.

    Composes a :class:`VggtGeometryTransformer` for model inference and a
    :class:`MultiViewTracker` for feature tracking and GtsfmData assembly.
    Cluster-level bundle adjustment is handled by :func:`_run_cluster_ba`.
    """

    def __init__(
        self,
        # --- Geometry transformer ---
        geometry_transformer: VggtGeometryTransformer | None = None,
        # --- Tracker ---
        tracker: MultiViewTracker | None = None,
        # --- Cluster BA params ---
        ba_options: BundleAdjustmentOptions | None = None,
        pre_ba_max_reproj_error: float = 14.0,
        post_ba_max_reproj_error: float = 3.0,
        drop_camera_with_no_track: bool = False,
        # --- VGGT-specific operational params ---
        weights_path: Optional[str] = None,
        input_mode: str = "crop",
        save_processed_image: bool = False,
        seed: int = 42,
        model_cache_key: Hashable | bool | None = None,
        metric_constructed_only: bool = False,
        # --- Base class params (output routing) ---
        output_worker: Optional[str] = None,
    ) -> None:
        super().__init__(output_worker=output_worker)

        self._weights_path = Path(weights_path) if weights_path is not None else None
        self._input_mode = input_mode
        self._save_processed_image = save_processed_image
        self._seed = seed
        self._metric_constructed_only = metric_constructed_only

        # --- Geometry transformer ---
        self.geometry_transformer = geometry_transformer or VggtGeometryTransformer()

        # --- Tracker ---
        self.tracker = tracker or MultiViewTracker()

        # --- Cluster BA params ---
        self.ba_options = ba_options or BundleAdjustmentOptions()
        self._pre_ba_max_reproj_error = pre_ba_max_reproj_error
        self._post_ba_max_reproj_error = post_ba_max_reproj_error
        self._drop_camera_with_no_track = drop_camera_with_no_track

        # --- Model caching ---
        self._loader_kwargs: dict[str, Any] = {}
        if self._weights_path is not None:
            self._loader_kwargs["weights_path"] = self._weights_path
        model_kwargs = self.geometry_transformer.config.model_ctor_kwargs
        if model_kwargs:
            self._loader_kwargs["model_kwargs"] = model_kwargs

        if model_cache_key is False:
            self._model_cache_key: Hashable | None = None
        elif model_cache_key is None:
            kwargs_key = (
                tuple(sorted((k, repr(v)) for k, v in model_kwargs.items()))
                if model_kwargs
                else None
            )
            self._model_cache_key = ("default_vggt_loader", self._weights_path, kwargs_key)
        else:
            self._model_cache_key = model_cache_key

    def __repr__(self) -> str:
        components = [
            f"geometry_transformer={self.geometry_transformer.config}",
            f"tracker={self.tracker.config}",
            f"ba_options={self.ba_options}",
            f"weights_path={self._weights_path}",
            f"input_mode={self._input_mode}",
        ]
        return "ClusterVGGT(\n  " + ",\n  ".join(str(c) for c in components) + "\n)"

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""
        return UiMetadata(
            display_name="VGGT",
            input_products=("Key Images",),
            output_products=("VGGT Reconstruction",),
            parent_plate="Cluster Optimizer",
        )

    def create_computation_graph(
        self,
        context: ClusterContext,
    ) -> ClusterComputationGraph | None:
        """Create the VGGT computation graph for a cluster."""

        keys = sorted(visibility_graph_keys(context.visibility_graph))
        if not keys:
            return None

        global_indices = tuple(int(idx) for idx in keys)
        image_filenames = context.loader.image_filenames()
        image_names = tuple(str(image_filenames[idx]) for idx in keys)
        dataset_dir = getattr(context.loader, "_dataset_dir", None)
        dataset_dir = str(dataset_dir) if dataset_dir is not None else None

        # 1. Load images.
        image_batch_graph, original_coords_graph = delayed(_load_vggt_inputs, nout=2)(
            context.loader,
            global_indices,
            mode=self._input_mode,
            save_processed_image=self._save_processed_image,
            output_root=str(context.output_paths.results),
            image_names=image_names,
        )

        # 2. Run VGGT pipeline (geometry + tracking -> GtsfmData, NO BA).
        pre_ba_data_graph = delayed(_run_vggt_pipeline)(
            image_batch_graph,
            original_coords_graph,
            transformer=self.geometry_transformer,
            tracker=self.tracker,
            image_indices=global_indices,
            image_names=image_names,
            dataset_dir=dataset_dir,
            seed=self._seed,
            model_cache_key=self._model_cache_key,
            loader_kwargs=self._loader_kwargs or None,
            weights_path=self._weights_path,
            cluster_label=context.label,
        )

        # 3. Run cluster-level BA.
        ba_result_graph, pre_ba_result_graph = delayed(_run_cluster_ba, nout=2)(
            pre_ba_data_graph,
            ba_options=self.ba_options,
            pre_ba_max_reproj_error=self._pre_ba_max_reproj_error,
            post_ba_max_reproj_error=self._post_ba_max_reproj_error,
            drop_camera_with_no_track=self._drop_camera_with_no_track,
            min_track_length=self.tracker.config.min_track_length,
            cluster_label=context.label,
        )

        # 4. Metrics.
        cameras_gt = [context.one_view_data_dict[idx].camera_gt for idx in range(context.num_images)]
        metrics_tasks = [
            delayed(_aggregate_vggt_metrics)(
                ba_result_graph,
                cameras_gt=cameras_gt,
                pre_ba_result=pre_ba_result_graph,
                save_dir=str(context.output_paths.metrics),
                metric_constructed_only=self._metric_constructed_only,
            )
        ]

        # 5. I/O tasks.
        io_tasks: list[Delayed] = []
        with self._output_annotation():
            io_tasks.append(
                delayed(_save_reconstruction_as_text)(
                    ba_result_graph,
                    context.output_paths.results,
                )
            )
            io_tasks.append(
                delayed(_save_pre_ba_reconstruction_as_text)(
                    pre_ba_result_graph,
                    context.output_paths.results,
                )
            )

        return ClusterComputationGraph(
            io_tasks=tuple(io_tasks),
            metric_tasks=tuple(metrics_tasks),
            sfm_result=ba_result_graph
        )
