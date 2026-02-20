"""VGGT-based cluster optimizer leveraging the demo VGGT pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, Optional, Union

import numpy as np
import torch
from dask.delayed import Delayed, delayed
from gtsam import Pose3

import gtsfm.common.types as gtsfm_types
import gtsfm.frontend.vggt as vggt
import gtsfm.utils.metrics as metrics_utils
from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterComputationGraph, ClusterContext, ClusterOptimizerBase
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.vggt import VggtConfiguration, VggtReconstruction
from gtsfm.products.visibility_graph import visibility_graph_keys
from gtsfm.ui.gtsfm_process import UiMetadata
from gtsfm.utils.logger import get_logger

logger = get_logger()

# Module-level cache to avoid reloading VGGT weights per cluster.
_VGGT_MODEL_CACHE: dict[Hashable, Any] = {}


def _load_vggt_inputs(loader, indices: list[int], mode: str):
    """Load and preprocess a batch of images for VGGT."""
    return vggt.load_image_batch_vggt_loader(loader, indices, mode=mode)


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
) -> VggtReconstruction:
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
    return vggt.run_reconstruction(image_batch, **kwargs)


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
    pre_ba_result: Optional[GtsfmData],
    results_path: Path,
) -> None:
    if pre_ba_result is None:
        return
    _save_reconstruction_as_text(pre_ba_result, results_path, subdir="vggt_pre_ba")


def _get_pose_metrics(
    result_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    save_dir: Optional[str] = None,
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
    )


def _aggregate_vggt_metrics(
    result: GtsfmData,
    cameras_gt: Optional[list[Optional[gtsfm_types.CAMERA_TYPE]]] = None,
    pre_ba_result: Optional[GtsfmData] = None,
    *,
    save_dir: Optional[str] = None,
) -> list[GtsfmMetricsGroup]:
    def _build_metrics_group(scene: GtsfmData, name: str) -> GtsfmMetricsGroup:
        metrics_group = GtsfmMetricsGroup(
            name,
            [
                GtsfmMetric("num_cameras", len(scene.get_valid_camera_indices())),
                GtsfmMetric("num_points3d", scene.number_tracks()),
            ],
        )
        if cameras_gt is not None:
            metrics_group.extend(_get_pose_metrics(scene, cameras_gt, save_dir=save_dir))
        return metrics_group

    metrics_groups = [_build_metrics_group(result, "cluster_vggt_metrics")]
    if pre_ba_result is not None:
        metrics_groups.append(_build_metrics_group(pre_ba_result, "cluster_vggt_pre_ba_metrics"))
    return metrics_groups


def _extract_post_ba_result(result: VggtReconstruction) -> GtsfmData:
    """Extract the post-BA reconstruction from the VGGT pipeline output."""
    return result.gtsfm_data


def _extract_pre_ba_result(result: VggtReconstruction) -> Optional[GtsfmData]:
    """Extract the optional pre-BA reconstruction for debugging."""
    return result.pre_ba_data


class ClusterVGGT(ClusterOptimizerBase):
    """Cluster optimizer that runs VGGT to generate COLMAP-style reconstructions."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 5.0,
        max_num_points: int = 100000,
        tracking: bool = False,
        tracking_max_query_pts: int = 2048,
        tracking_query_frame_num: int = 3,
        track_vis_thresh: float = 0.05,
        track_conf_thresh: float = 0.2,
        keypoint_extractor: str = "aliked+sp+sift",
        camera_type: str = "PINHOLE",
        seed: int = 42,
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
        store_pre_ba_result: bool = False,
        run_bundle_adjustment_on_parent: bool = True,
        max_reproj_error: float = 8.0,
        min_triangulation_angle: float = 10.0,
        plot_reprojection_histograms: bool = True,
        merge_duplicate_tracks: bool = True,
        drop_outlier_after_camera_merging: bool = True,
        drop_child_if_merging_fail: bool = True,
        drop_camera_with_no_track: bool = True,
        ba_use_calibration_prior: bool = False,
        ba_use_undistorted_camera_model: bool = False,
        use_shared_calibration: bool = True,
    ) -> None:
        super().__init__(
            pose_angular_error_thresh=pose_angular_error_thresh,
            output_worker=output_worker,
            drop_child_if_merging_fail=drop_child_if_merging_fail,
            drop_camera_with_no_track=drop_camera_with_no_track,
            drop_outlier_after_camera_merging=drop_outlier_after_camera_merging,
            plot_reprojection_histograms=plot_reprojection_histograms,
            run_bundle_adjustment_on_parent=run_bundle_adjustment_on_parent,
            use_shared_calibration=use_shared_calibration,
            merge_duplicate_tracks=merge_duplicate_tracks,
        )
        self._weights_path = Path(weights_path) if weights_path is not None else None
        self._conf_threshold = conf_threshold
        self._max_points_for_colmap = max_num_points
        self._tracking = tracking
        self._tracking_max_query_pts = tracking_max_query_pts
        self._tracking_query_frame_num = tracking_query_frame_num
        self._track_vis_thresh = track_vis_thresh
        self._track_conf_thresh = track_conf_thresh
        self._keypoint_extractor = keypoint_extractor
        self._camera_type = camera_type
        self._max_reproj_error = max_reproj_error
        self._min_triangulation_angle = min_triangulation_angle
        self._seed = seed
        self._explicit_scene_dir = Path(scene_dir) if scene_dir is not None else None
        self._use_sparse_attention = use_sparse_attention
        self._dtype = inference_dtype
        self._run_bundle_adjustment_on_leaf = run_bundle_adjustment_on_leaf
        self._store_pre_ba_result = store_pre_ba_result
        self._min_triangulation_angle = min_triangulation_angle
        self._ba_use_calibration_prior = ba_use_calibration_prior
        self._ba_use_undistorted_camera_model = ba_use_undistorted_camera_model
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
            confidence_threshold=self._conf_threshold,
            max_num_points=self._max_points_for_colmap,
            tracking=self._tracking,
            max_query_pts=self._tracking_max_query_pts,
            query_frame_num=self._tracking_query_frame_num,
            track_vis_thresh=self._track_vis_thresh,
            track_conf_thresh=self._track_conf_thresh,
            keypoint_extractor=self._keypoint_extractor,
            dtype=self._dtype,
            model_ctor_kwargs=self._model_ctor_kwargs.copy(),
            use_sparse_attention=self._use_sparse_attention,
            run_bundle_adjustment_on_leaf=self._run_bundle_adjustment_on_leaf,
            store_pre_ba_result=self._store_pre_ba_result,
            max_reproj_error=self._max_reproj_error,
            min_triangulation_angle=self._min_triangulation_angle,
            ba_use_calibration_prior=self._ba_use_calibration_prior,
            ba_use_undistorted_camera_model=self._ba_use_undistorted_camera_model,
            ba_use_shared_calibration=self.use_shared_calibration,
        )

        # mode is fixed to "crop", it resizes the width to 518 while maintaining aspect ratio and only if
        # height is > 518 then crops
        image_batch_graph, original_coords_graph = delayed(_load_vggt_inputs, nout=2)(
            context.loader, global_indices, mode="crop"
        )

        reconstruction_graph = delayed(_run_vggt_pipeline)(
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
        result_graph = delayed(_extract_post_ba_result)(reconstruction_graph)
        pre_ba_result_graph = delayed(_extract_pre_ba_result)(reconstruction_graph)

        cameras_gt = [context.one_view_data_dict[idx].camera_gt for idx in range(context.num_images)]
        metrics_tasks = [
            delayed(_aggregate_vggt_metrics)(
                result_graph,
                cameras_gt=cameras_gt,
                pre_ba_result=pre_ba_result_graph,
                save_dir=str(context.output_paths.metrics),
            )
        ]

        io_tasks: list[Delayed] = []
        with self._output_annotation():
            io_tasks.append(
                delayed(_save_reconstruction_as_text)(
                    result_graph,
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
            sfm_result=result_graph,
        )
