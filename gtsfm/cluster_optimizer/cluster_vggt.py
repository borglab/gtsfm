"""VGGT-based cluster optimizer leveraging the demo VGGT pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from dask.delayed import Delayed, delayed

import gtsfm.frontend.vggt as vggt
from gtsfm.cluster_optimizer.cluster_optimizer_base import (
    REACT_RESULTS_PATH,
    ClusterComputationGraph,
    ClusterContext,
    ClusterOptimizerBase,
)
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.vggt import VggtConfiguration
from gtsfm.products.visibility_graph import visibility_graph_keys
from gtsfm.ui.gtsfm_process import UiMetadata
from gtsfm.utils.logger import get_logger

logger = get_logger()

# Module-level cache to avoid reloading VGGT weights per cluster.
_VGGT_MODEL_CACHE: dict[Hashable, Any] = {}


def _resize_to_square_tensor(image: np.ndarray, target_size: int) -> torch.Tensor:
    """Resize a HxWx3 numpy image to a square torch tensor normalized to [0,1]."""
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return (tensor.squeeze(0)) / 255.0


def _load_vggt_inputs(loader, indices: list[int], target_size: int):
    """Load and preprocess a batch of images for VGGT."""

    def resize_transform(arr: np.ndarray) -> torch.Tensor:
        return _resize_to_square_tensor(arr, target_size)

    return loader.load_image_batch_vggt(indices, target_size, resize_transform)


def _resolve_vggt_model(cache_key: Hashable | None, loader_kwargs: dict[str, Any] | None) -> Any | None:
    """Fetch (or lazily load) a VGGT model for the current worker."""

    if cache_key is None:
        return None

    if cache_key in _VGGT_MODEL_CACHE:
        return _VGGT_MODEL_CACHE[cache_key]

    logger.info("⏳ Loading VGGT model weights...")
    loader_kwargs = loader_kwargs or {}
    model = vggt.load_model(**loader_kwargs)
    _VGGT_MODEL_CACHE[cache_key] = model
    logger.info("✅ VGGT model weights loaded successfully.")
    return model


def _run_vggt_pipeline(
    image_batch: torch.Tensor,
    seed: int,
    *,
    model_cache_key: Hashable | None = None,
    loader_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> GtsfmData:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cluster_label = kwargs.pop("cluster_label", None)
    partition_indices = kwargs.get("image_indices")
    if cluster_label is not None:
        logger.info("🔵 Running VGGT on %s with %d images.", str(cluster_label).lower(), image_batch.shape[0])
    elif partition_indices:
        logger.info("🔵 Running VGGT on %d images for partition %s.", image_batch.shape[0], partition_indices)
    else:
        logger.info("🔵 Running VGGT on %d images.", image_batch.shape[0])
    cached_model = _resolve_vggt_model(model_cache_key, loader_kwargs)
    if cached_model is not None:
        kwargs = {**kwargs, "model": cached_model}
    return vggt.run_reconstruction_gtsfm_data_only(image_batch, **kwargs)


def _save_reconstruction_as_text(
    result: GtsfmData,
    results_path: Path,
    copy_to_react: bool,
    relative_results_dir: Path,
) -> None:
    target_dir = results_path / "vggt"
    target_dir.mkdir(parents=True, exist_ok=True)
    result.export_as_colmap_text(target_dir)

    if not copy_to_react:
        return

    react_destination = REACT_RESULTS_PATH / relative_results_dir / "vggt"
    react_destination.mkdir(parents=True, exist_ok=True)
    result.export_as_colmap_text(react_destination)


def _aggregate_vggt_metrics(result: GtsfmData) -> GtsfmMetricsGroup:
    num_cameras = len(result.get_valid_camera_indices())
    num_points3d = result.number_tracks()
    return GtsfmMetricsGroup(
        "vggt_runtime_metrics",
        [
            GtsfmMetric("num_cameras", num_cameras),
            GtsfmMetric("num_points3d", num_points3d),
        ],
    )


class ClusterVGGT(ClusterOptimizerBase):
    """Cluster optimizer that runs VGGT to generate COLMAP-style reconstructions."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        image_load_resolution: int = 1024,
        inference_resolution: int = 518,
        conf_threshold: float = 5.0,
        max_num_points: int = 100000,
        tracking: bool = False,
        tracking_max_query_pts: int = 1000,
        tracking_query_frame_num: int = 4,
        tracking_fine_tracking: bool = True,
        track_vis_thresh: float = 0.2,
        camera_type: str = "PINHOLE",
        seed: int = 42,
        copy_results_to_react: bool = True,
        scene_dir: Optional[str] = None,
        pose_angular_error_thresh: float = 3.0,
        output_worker: Optional[str] = None,
        model_cache_key: Hashable | bool | None = None,
        inference_dtype: Optional[Union[str, torch.dtype]] = None,
        model_ctor_kwargs: Optional[dict[str, Any]] = None,
        use_sparse_attention: bool = False,
        fast_dtype: Optional[Union[str, torch.dtype]] = None,
        merging: Optional[int] = None,
        vis_attn_map: bool = False,
        enable_protection: bool = False,
        extra_model_kwargs: Optional[dict[str, Any]] = None,
        run_bundle_adjustment_on_leaf: bool = False,
        run_bundle_adjustment_on_parent: bool = True,
        max_reproj_error: float = 8.0,
        plot_reprojection_histograms: bool = True,
        drop_outlier_after_camera_merging: bool = True,
        drop_child_if_merging_fail: bool = True,
        drop_camera_with_no_track: bool = True,
    ) -> None:
        super().__init__(
            pose_angular_error_thresh=pose_angular_error_thresh,
            output_worker=output_worker,
        )
        self.plot_reprojection_histograms = plot_reprojection_histograms
        self.drop_outlier_after_camera_merging = drop_outlier_after_camera_merging
        self.drop_child_if_merging_fail = drop_child_if_merging_fail
        self.drop_camera_with_no_track = drop_camera_with_no_track
        self._weights_path = Path(weights_path) if weights_path is not None else None
        self._image_load_resolution = image_load_resolution
        self._inference_resolution = inference_resolution
        self._conf_threshold = conf_threshold
        self._max_points_for_colmap = max_num_points
        self._tracking = tracking
        self._tracking_max_query_pts = tracking_max_query_pts
        self._tracking_query_frame_num = tracking_query_frame_num
        self._tracking_fine_tracking = tracking_fine_tracking
        self._track_vis_thresh = track_vis_thresh
        self._camera_type = camera_type
        self._max_reproj_error = max_reproj_error
        self._seed = seed
        self._copy_results_to_react = copy_results_to_react
        self._explicit_scene_dir = Path(scene_dir) if scene_dir is not None else None
        self._use_sparse_attention = use_sparse_attention
        self._dtype = inference_dtype
        self._run_bundle_adjustment_on_leaf = run_bundle_adjustment_on_leaf
        if fast_dtype is not None:
            if self._dtype is None:
                self._dtype = fast_dtype
            elif self._dtype != fast_dtype:
                logger.warning(
                    "Ignoring fast_dtype=%s because inference_dtype=%s is already specified.",
                    fast_dtype,
                    self._dtype,
                )

        self._model_ctor_kwargs = dict(model_ctor_kwargs) if model_ctor_kwargs is not None else {}
        if extra_model_kwargs:
            self._model_ctor_kwargs.update(extra_model_kwargs)

        def _maybe_set_model_kw(key: str, value: Any) -> None:
            if value is None:
                return
            self._model_ctor_kwargs.setdefault(key, value)

        _maybe_set_model_kw("merging", merging)
        if vis_attn_map:
            self._model_ctor_kwargs.setdefault("vis_attn_map", True)
        if enable_protection:
            self._model_ctor_kwargs.setdefault("enable_protection", True)

        self._loader_kwargs: dict[str, Any] = {}
        if self._weights_path is not None:
            self._loader_kwargs["weights_path"] = self._weights_path
        if self._model_ctor_kwargs:
            self._loader_kwargs["model_kwargs"] = self._model_ctor_kwargs
        self.run_bundle_adjustment_on_parent = run_bundle_adjustment_on_parent

        if model_cache_key is False:
            self._model_cache_key: Hashable | None = None
        elif model_cache_key is None:
            kwargs_key = (
                tuple(sorted((k, repr(v)) for k, v in self._model_ctor_kwargs.items()))
                if self._model_ctor_kwargs
                else None
            )
            self._model_cache_key = ("default_vggt_loader", self._weights_path, kwargs_key)
        else:
            self._model_cache_key = model_cache_key

    def __repr__(self) -> str:
        components = [
            f"weights_path={self._weights_path}",
            f"image_load_resolution={self._image_load_resolution}",
            f"inference_resolution={self._inference_resolution}",
            f"camera_type={self._camera_type}",
            f"dtype={self._dtype}",
            f"use_sparse_attention={self._use_sparse_attention}",
            f"run_bundle_adjustment_on_leaf={self._run_bundle_adjustment_on_leaf}",
        ]
        if self._model_ctor_kwargs:
            components.append(f"model_ctor_kwargs={self._model_ctor_kwargs}")
        return "ClusterVGGT(\n  " + ",\n  ".join(str(c) for c in components) + "\n)"

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""
        # This class and its subclasses (unless overridden) are not part of the UI.
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

        config = VggtConfiguration(
            vggt_fixed_resolution=self._inference_resolution,
            img_load_resolution=self._image_load_resolution,
            confidence_threshold=self._conf_threshold,
            max_num_points=self._max_points_for_colmap,
            tracking=self._tracking,
            max_query_pts=self._tracking_max_query_pts,
            query_frame_num=self._tracking_query_frame_num,
            fine_tracking=self._tracking_fine_tracking,
            track_vis_thresh=self._track_vis_thresh,
            dtype=self._dtype,
            model_ctor_kwargs=self._model_ctor_kwargs.copy(),
            use_sparse_attention=self._use_sparse_attention,
            run_bundle_adjustment_on_leaf=self._run_bundle_adjustment_on_leaf,
            max_reproj_error=self._max_reproj_error,
        )

        image_batch_graph, original_coords_graph = delayed(_load_vggt_inputs, nout=2)(
            context.loader, global_indices, self._image_load_resolution
        )

        result_graph = delayed(_run_vggt_pipeline)(
            image_batch_graph,
            seed=self._seed,
            original_coords=original_coords_graph,
            image_indices=global_indices,
            image_names=image_names,
            config=config,
            weights_path=self._weights_path,
            model_cache_key=self._model_cache_key,
            loader_kwargs=self._loader_kwargs or None,
            cluster_label=context.label,
        )

        metrics_tasks = [delayed(_aggregate_vggt_metrics)(result_graph)]

        io_tasks: list[Delayed] = []
        with self._output_annotation():
            io_tasks.append(
                delayed(_save_reconstruction_as_text)(
                    result_graph,
                    context.output_paths.results,
                    self._copy_results_to_react,
                    context.react_results_subdir,
                )
            )

        return ClusterComputationGraph(
            io_tasks=tuple(io_tasks),
            metric_tasks=tuple(metrics_tasks),
            sfm_result=result_graph,
        )
