"""Cluster optimizer that combines a traditional MVO frontend with VGGT geometry + BA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, NamedTuple, Optional

import numpy as np
import torch
from dask.delayed import delayed
from gtsam import Point2, Point3, SfmTrack

import gtsfm.common.types as gtsfm_types
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptions
from gtsfm.cluster_optimizer.cluster_mvo import ClusterMVO
from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterComputationGraph, ClusterContext
from gtsfm.cluster_optimizer.cluster_vggt import (
    _aggregate_vggt_metrics,
    _load_vggt_inputs,
    _resolve_vggt_model,
    _run_cluster_ba,
    _save_pre_ba_reconstruction_as_text,
    _save_reconstruction_as_text,
)
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.vggt_geometry_transformer import VggtGeometryTransformer
from gtsfm.multi_view_optimizer import get_2d_tracks
from gtsfm.products.visibility_graph import visibility_graph_keys
from gtsfm.two_view_estimator import TwoViewEstimator
from gtsfm.ui.gtsfm_process import UiMetadata
from gtsfm.view_graph_estimator.view_graph_calibration import calibrate_view_graph
import gtsam

from gtsfm.utils import torch as torch_utils
from gtsfm.utils.logger import get_logger

logger = get_logger()


class VggtGeometryResult(NamedTuple):
    """Outputs of VGGT geometry prediction needed for downstream processing."""

    cameras: dict[int, gtsfm_types.CAMERA_TYPE]
    dense_points: np.ndarray       # (N, H, W, 3) world-space 3D points per pixel
    depth_confidence: np.ndarray   # (N, H, W) per-pixel depth confidence scores
    original_coords: np.ndarray    # (N, 6) VGGT crop/pad metadata


def _extract_v_corr_idxs(two_view_results) -> dict:
    """Pull verified correspondence index arrays out of two-view results."""
    return {ij: result.v_corr_idxs for ij, result in two_view_results.items()}


def _identity(x):
    """Wrap an eager value as a single delayed node (so it is embedded once, not per consumer)."""
    return x


def _get_image_shapes(loader, image_indices: tuple[int, ...]) -> dict[int, tuple[int, int]]:
    """Return original (height, width) for each requested image index."""
    return {idx: loader.get_image(idx).value_array.shape[:2] for idx in image_indices}


def _run_vggt_geometry(
    image_batch: torch.Tensor,
    original_coords: torch.Tensor,
    *,
    transformer: VggtGeometryTransformer,
    image_indices: tuple[int, ...],
    seed: int = 42,
    model_cache_key: Hashable | None = None,
    loader_kwargs: dict[str, Any] | None = None,
    weights_path: Optional[Path] = None,
    cluster_label: Optional[str] = None,
) -> VggtGeometryResult:
    """Run VGGT geometry prediction, returning cameras and dense 3D points."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if cluster_label:
        logger.info("🔵 Running VGGT geometry on %s with %d images.", cluster_label, image_batch.shape[0])

    cached_model = _resolve_vggt_model(model_cache_key, loader_kwargs)
    geo_output = transformer.predict(image_batch, model=cached_model, weights_path=weights_path)

    original_coords_np = original_coords.to(torch.float32).cpu().numpy()
    cameras = {
        global_idx: torch_utils.camera_from_matrices(
            geo_output.extrinsic[local_idx].to(torch.float32).cpu().numpy(),
            geo_output.intrinsic[local_idx].to(torch.float32).cpu().numpy(),
            crop_coords=original_coords_np[local_idx],
            use_cal3_bundler=True,
        )
        for local_idx, global_idx in enumerate(image_indices)
    }

    result = VggtGeometryResult(
        cameras=cameras,
        dense_points=geo_output.dense_points.to(torch.float32).cpu().numpy(),
        depth_confidence=geo_output.depth_confidence.to(torch.float32).cpu().numpy(),
        original_coords=original_coords_np,
    )

    if geo_output.device.type == "cuda":
        del geo_output
        torch.cuda.empty_cache()

    return result


def _scale_camera_intrinsics(
    camera: gtsfm_types.CAMERA_TYPE, scale: float
) -> gtsfm_types.CAMERA_TYPE:
    """Return a copy of camera with intrinsics uniformly scaled, pose unchanged."""
    pose = camera.pose()
    cal = camera.calibration()
    if isinstance(cal, gtsam.Cal3Bundler):
        return gtsam.PinholeCameraCal3Bundler(
            pose,
            gtsam.Cal3Bundler(cal.fx() * scale, cal.k1(), cal.k2(), cal.px() * scale, cal.py() * scale),
        )
    if isinstance(cal, gtsam.Cal3_S2):
        return gtsam.PinholeCameraCal3_S2(
            pose,
            gtsam.Cal3_S2(cal.fx() * scale, cal.fy() * scale, 0.0, cal.px() * scale, cal.py() * scale),
        )
    raise ValueError(f"Unsupported calibration type: {type(cal)}")


def _build_gtsfm_data_from_vggt_depth(
    vggt_result: VggtGeometryResult,
    tracks_2d: list[SfmTrack2d],
    image_shapes: dict[int, tuple[int, int]],
    image_indices: tuple[int, ...],
    num_images: int,
    min_track_length: int = 2,
    refined_intrinsics: Optional[dict[int, gtsfm_types.CALIBRATION_TYPE]] = None,
) -> GtsfmData:
    """Build GtsfmData using VGGT cameras (rescaled to original resolution) and frontend 2D tracks.

    VGGT camera intrinsics are scaled from VGGT pixel space to original image resolution so
    that the frontend keypoints (in original coords) can be used directly as BA measurements.
    VGGT pixel coordinates are used only to look up per-pixel depth values for 3D initialisation.

    Args:
        vggt_result: Cameras, dense 3D points, and crop metadata from VGGT.
        tracks_2d: 2D feature tracks from the frontend (in original image coordinates).
        image_shapes: Original (height, width) per global image index.
        image_indices: Ordered global image indices corresponding to the N VGGT frames.
        num_images: Total number of images in the scene.
        min_track_length: Minimum observations per track; shorter tracks are dropped.

    Returns:
        GtsfmData with intrinsics-rescaled VGGT cameras and depth-initialised tracks whose
        2D measurements are in the original keypoint coordinate system.
    """
    cameras = vggt_result.cameras
    dense_points = vggt_result.dense_points        # (N, H_vggt, W_vggt, 3)
    depth_confidence = vggt_result.depth_confidence  # (N, H_vggt, W_vggt)
    original_coords = vggt_result.original_coords  # (N, 6): [left, top, right, bottom, sw, sh]

    _, H_vggt, W_vggt = dense_points.shape[:3]
    global_to_local = {gidx: lidx for lidx, gidx in enumerate(image_indices)}

    # Register cameras with intrinsics in original image resolution. If
    # `refined_intrinsics` is supplied (from view-graph calibration), use those
    # directly; otherwise rescale VGGT's predicted intrinsics from VGGT pixel space.
    gtsfm_data = GtsfmData(number_images=num_images)
    for global_idx, camera in cameras.items():
        if global_idx in image_shapes and global_idx in global_to_local:
            if refined_intrinsics is not None:
                camera = type(camera)(camera.pose(), refined_intrinsics[global_idx])
            else:
                _, orig_W = image_shapes[global_idx]
                local_idx = global_to_local[global_idx]
                scaled_W = float(original_coords[local_idx, 4])
                camera = _scale_camera_intrinsics(camera, scale=orig_W / scaled_W)
        gtsfm_data.add_camera(global_idx, camera)

    for track_2d in tracks_2d:
        if track_2d.number_measurements() < min_track_length:
            continue

        points_3d: list[np.ndarray] = []
        confidences: list[float] = []
        valid_measurements: list[tuple[int, np.ndarray]] = []

        for m in track_2d.measurements:
            global_idx = m.i
            if global_idx not in global_to_local or global_idx not in image_shapes:
                continue

            local_idx = global_to_local[global_idx]
            orig_H, orig_W = image_shapes[global_idx]

            # Map frontend keypoints into VGGT dense-map coordinates using the actual
            # per-axis resized image dimensions plus the crop/pad offsets recorded in
            # original_coords for this image.
            left, top = original_coords[local_idx, 0], original_coords[local_idx, 1]
            scaled_W, scaled_H = original_coords[local_idx, 4], original_coords[local_idx, 5]
            u_scale = scaled_W / orig_W if orig_W > 0 else 0.0
            v_scale = scaled_H / orig_H if orig_H > 0 else 0.0
            u_c = int(np.clip(round(m.uv[0] * u_scale - left), 0, W_vggt - 1))
            v_c = int(np.clip(round(m.uv[1] * v_scale - top), 0, H_vggt - 1))

            pt3d = dense_points[local_idx, v_c, u_c]
            conf = float(depth_confidence[local_idx, v_c, u_c])
            if np.isfinite(pt3d).all() and conf > 0.0:
                points_3d.append(pt3d)
                confidences.append(conf)
                valid_measurements.append((global_idx, m.uv))  # original keypoint coords

        if len(points_3d) < min_track_length:
            continue

        weights = np.array(confidences)
        weights /= weights.sum()
        point_3d_mean = np.average(points_3d, axis=0, weights=weights)
        sfm_track = SfmTrack(Point3(*point_3d_mean.astype(float)))
        for gidx, uv in valid_measurements:
            if gidx in cameras:
                sfm_track.addMeasurement(gidx, Point2(*uv.astype(float)))

        if sfm_track.numberMeasurements() >= min_track_length:
            gtsfm_data.add_track(sfm_track)

    logger.info("Built GtsfmData with %d cameras and %d tracks.", len(cameras), gtsfm_data.number_tracks())
    return gtsfm_data


def _refine_vggt_intrinsics_via_view_graph(
    vggt_result: VggtGeometryResult,
    v_corr_idxs: dict,
    keypoints_list: list,
    image_shapes: dict[int, tuple[int, int]],
    image_indices: tuple[int, ...],
) -> dict[int, gtsfm_types.CALIBRATION_TYPE]:
    """Refine focal lengths via Fetzer joint optimization over F-matrix edges.

    Returns intrinsics in ORIGINAL image coordinates (suitable for use with the
    frontend keypoints). VGGT's predicted intrinsics are rescaled to original
    image coords first, then handed to `calibrate_view_graph` as the initial
    estimate. VGGT's predicted poses are unchanged — only intrinsics are refined.
    """
    initial_intrinsics: dict[int, gtsfm_types.CALIBRATION_TYPE] = {}
    for local_idx, global_idx in enumerate(image_indices):
        if global_idx in vggt_result.cameras and global_idx in image_shapes:
            _, orig_W = image_shapes[global_idx]
            scaled_W = float(vggt_result.original_coords[local_idx, 4])
            scale = orig_W / scaled_W if scaled_W > 0 else 1.0
            scaled_cam = _scale_camera_intrinsics(vggt_result.cameras[global_idx], scale=scale)
            initial_intrinsics[global_idx] = scaled_cam.calibration()

    keypoints = {gidx: keypoints_list[gidx] for gidx in image_indices}
    refined, _edges_to_remove = calibrate_view_graph(
        v_corr_idxs_dict=v_corr_idxs,
        keypoints=keypoints,
        initial_intrinsics=initial_intrinsics,
    )
    return refined


class ClusterVGGTWithFrontend(ClusterMVO):
    """Cluster optimizer that combines a traditional MVO frontend with VGGT poses.

    Camera poses come from VGGT geometry prediction. 2D feature tracks come from the
    frontend (correspondence generation + two-view estimation). Each track's 3D point is
    initialised from VGGT's dense depth map rather than via triangulation, then refined
    by cluster-level bundle adjustment.
    """

    def __init__(
        self,
        correspondence_generator: CorrespondenceGeneratorBase,
        two_view_estimator: TwoViewEstimator,
        geometry_transformer: VggtGeometryTransformer | None = None,
        ba_options: BundleAdjustmentOptions | None = None,
        # Cluster BA params
        pre_ba_max_reproj_error: float = 14.0,
        post_ba_max_reproj_error: float = 3.0,
        drop_camera_with_no_track: bool = False,
        min_track_length: int = 2,
        # VGGT model loading
        weights_path: Optional[str] = None,
        input_mode: str = "crop",
        seed: int = 42,
        model_cache_key: Hashable | bool | None = None,
        metric_constructed_only: bool = False,
        # Frontend params
        save_two_view_viz: bool = False,
        pose_angular_error_thresh: float = 3,
        output_worker: Optional[str] = None,
        use_view_graph_calibration: bool = False,
        use_multi_view_retriangulation: bool = False,
    ) -> None:
        super().__init__(
            correspondence_generator=correspondence_generator,
            two_view_estimator=two_view_estimator,
            multiview_optimizer=None,  # VGGT depth replaces triangulation + MVO
            save_two_view_viz=save_two_view_viz,
            pose_angular_error_thresh=pose_angular_error_thresh,
            output_worker=output_worker,
        )
        self.geometry_transformer = geometry_transformer or VggtGeometryTransformer()
        self.ba_options = ba_options or BundleAdjustmentOptions()
        self._pre_ba_max_reproj_error = pre_ba_max_reproj_error
        self._post_ba_max_reproj_error = post_ba_max_reproj_error
        self._drop_camera_with_no_track = drop_camera_with_no_track
        self._min_track_length = min_track_length
        self._metric_constructed_only = metric_constructed_only
        self._input_mode = input_mode
        self._seed = seed
        self._use_view_graph_calibration = use_view_graph_calibration
        self._use_multi_view_retriangulation = use_multi_view_retriangulation

        self._weights_path = Path(weights_path) if weights_path is not None else None
        self._loader_kwargs: dict[str, Any] = {}
        if self._weights_path is not None:
            self._loader_kwargs["weights_path"] = self._weights_path
        model_kwargs = self.geometry_transformer.config.model_ctor_kwargs
        if model_kwargs:
            self._loader_kwargs["model_kwargs"] = model_kwargs

        if model_cache_key is False:
            self._model_cache_key: Hashable | None = None
        elif model_cache_key is None:
            kwargs_key = tuple(sorted((k, repr(v)) for k, v in model_kwargs.items())) if model_kwargs else None
            self._model_cache_key = ("default_vggt_loader", self._weights_path, kwargs_key)
        else:
            self._model_cache_key = model_cache_key

    def __repr__(self) -> str:
        components = [
            f"correspondence_generator={self.correspondence_generator}",
            f"two_view_estimator={self.two_view_estimator}",
            f"geometry_transformer={self.geometry_transformer.config}",
            f"ba_options={self.ba_options}",
        ]
        return "ClusterVGGTWithFrontend(\n  " + ",\n  ".join(components) + "\n)"

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        return UiMetadata(
            display_name="VGGT + Frontend",
            input_products=("Key Images",),
            output_products=("VGGT Reconstruction",),
            parent_plate="Cluster Optimizer",
        )

    def create_computation_graph(self, context: ClusterContext) -> ClusterComputationGraph | None:
        keys = sorted(visibility_graph_keys(context.visibility_graph))
        if not keys:
            return None

        global_indices = tuple(int(idx) for idx in keys)
        image_filenames = context.loader.image_filenames()
        image_names = tuple(str(image_filenames[idx]) for idx in keys)

        # This cluster's 2D tracks. When the context carries the globally-verified correspondences,
        # subset them to this cluster's edges and build the tracks EAGERLY in the main process — the
        # per-cluster frontend would recompute the identical per-edge result (same edges, same two-view
        # estimator), serially and redundantly across overlapping clusters. Without them (optimizer used
        # standalone), run the traditional per-cluster frontend.
        if context.global_v_corr_idxs_dict is not None and context.global_keypoints is not None:
            cluster_v_corr = {
                ij: context.global_v_corr_idxs_dict[ij]
                for ij in context.visibility_graph
                if ij in context.global_v_corr_idxs_dict
            }
            tracks_2d = get_2d_tracks(cluster_v_corr, context.global_keypoints)
            logger.info(
                "♻️  [%s] Reusing global correspondences: %d/%d cluster edges → %d tracks (frontend skipped).",
                context.label,
                len(cluster_v_corr),
                len(context.visibility_graph),
                len(tracks_2d),
            )
            v_corr_idxs_graph = delayed(_identity)(cluster_v_corr)
            tracks_2d_graph = delayed(_identity)(tracks_2d)
            padded_keypoints_graph = delayed(_identity)(context.global_keypoints)
            io_tasks, metrics = [], []
        else:
            frontend_graphs = self._build_frontend_graphs(context)
            io_tasks, metrics = self._build_frontend_output_graphs(context, frontend_graphs)
            v_corr_idxs_graph = delayed(_extract_v_corr_idxs)(frontend_graphs.two_view_results)
            tracks_2d_graph = delayed(get_2d_tracks)(v_corr_idxs_graph, frontend_graphs.padded_keypoints)
            padded_keypoints_graph = frontend_graphs.padded_keypoints

        # VGGT geometry prediction → cameras + dense 3D points.
        image_batch_graph, original_coords_graph = delayed(_load_vggt_inputs, nout=2)(
            context.loader,
            global_indices,
            mode=self._input_mode,
            output_root=str(context.output_paths.results),
            image_names=image_names,
        )
        vggt_result_graph = delayed(_run_vggt_geometry)(
            image_batch_graph,
            original_coords_graph,
            transformer=self.geometry_transformer,
            image_indices=global_indices,
            seed=self._seed,
            model_cache_key=self._model_cache_key,
            loader_kwargs=self._loader_kwargs or None,
            weights_path=self._weights_path,
            cluster_label=context.label,
        )

        # Original image shapes (needed to map frontend pixel coords → VGGT pixel coords).
        image_shapes_graph = delayed(_get_image_shapes)(context.loader, global_indices)

        # Optional: refine VGGT's predicted intrinsics via Fetzer joint
        # optimization over the frontend's F-matrices (keeps VGGT's predicted poses).
        refined_intrinsics_graph = None
        if self._use_view_graph_calibration:
            refined_intrinsics_graph = delayed(_refine_vggt_intrinsics_via_view_graph)(
                vggt_result_graph,
                v_corr_idxs_graph,
                padded_keypoints_graph,
                image_shapes_graph,
                global_indices,
            )

        # Build GtsfmData: lift 2D tracks to 3D using VGGT depth map.
        ba_input_graph = delayed(_build_gtsfm_data_from_vggt_depth)(
            vggt_result_graph,
            tracks_2d_graph,
            image_shapes=image_shapes_graph,
            image_indices=global_indices,
            num_images=context.num_images,
            min_track_length=self._min_track_length,
            refined_intrinsics=refined_intrinsics_graph,
        )

        # Cluster-level BA.
        ba_result_graph, pre_ba_result_graph = delayed(_run_cluster_ba, nout=2)(
            ba_input_graph,
            ba_options=self.ba_options,
            pre_ba_max_reproj_error=self._pre_ba_max_reproj_error,
            post_ba_max_reproj_error=self._post_ba_max_reproj_error,
            drop_camera_with_no_track=self._drop_camera_with_no_track,
            min_track_length=self._min_track_length,
            cluster_label=context.label,
            tracks_2d=tracks_2d_graph,
            use_multi_view_retriangulation=self._use_multi_view_retriangulation,
        )

        # Metrics + I/O.
        cameras_gt = [context.one_view_data_dict[idx].camera_gt for idx in range(context.num_images)]
        metrics.append(
            delayed(_aggregate_vggt_metrics)(
                ba_result_graph,
                cameras_gt=cameras_gt,
                pre_ba_result=pre_ba_result_graph,
                save_dir=str(context.output_paths.metrics),
                metric_constructed_only=self._metric_constructed_only,
            )
        )
        with self._output_annotation():
            io_tasks.append(delayed(_save_reconstruction_as_text)(ba_result_graph, context.output_paths.results))
            io_tasks.append(
                delayed(_save_pre_ba_reconstruction_as_text)(pre_ba_result_graph, context.output_paths.results)
            )

        return ClusterComputationGraph(
            io_tasks=tuple(io_tasks),
            metric_tasks=tuple(metrics),
            sfm_result=ba_result_graph,
        )
