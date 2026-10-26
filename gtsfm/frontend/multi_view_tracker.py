"""Multi-view tracker that uses geometry transformer output to produce tracks and GtsfmData."""

from __future__ import annotations

import os
import sqlite3
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from PIL import Image as PILImage
import numpy as np
import torch
import torch.nn.functional as F
from gtsam import Point2, Point3
from torch.amp import autocast as amp_autocast  # type: ignore
import yaml

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.frontend.geometry_transformer import GeometryTransformerOutput
from gtsfm.utils import logger as logger_utils
from gtsfm.utils import torch as torch_utils

PathLike = Union[str, Path]

logger = logger_utils.get_logger()

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Per-worker cache for DINOv2 model to avoid reloading per cluster/task.
_DINO_V2_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_DEFAULT_VGGT_COLMAPTRACKER_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "vggt_colmaptracker.yaml"
)
_COLMAP_TRACKER_TARGET_SIZE = 518
_COLMAP_MAX_IMAGE_ID = 2147483647


@dataclass(frozen=True)
class _CachedColmapDatabase:
    """Cached metadata read once from a COLMAP database."""

    db_path: Path
    image_id_by_name: dict[int, str]
    name_to_image_id: dict[str, int]
    keypoints_by_image_id: dict[int, np.ndarray]
    matches: list[tuple[int, int, np.ndarray]]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrackingConfig:
    """Configuration for multi-view tracking and GtsfmData assembly."""

    tracking: bool = True
    max_query_pts: int = 2048
    query_frame_num: int = 3
    use_all_frames_forward_only: bool = False
    track_vis_thresh: float = 0.05
    track_conf_thresh: float = 0.2
    keypoint_extractor: str = "aliked+sp+sift"
    vggt_max_reproj_error: float = 14.0
    min_triangulation_angle: float = 10.0
    min_track_length: int = 2
    ba_track_patch_size: int = 8
    enable_ba_track_patching: bool = False
    ba_use_undistorted_camera_model: bool = False
    colmaptracker_config_path: str = str(_DEFAULT_VGGT_COLMAPTRACKER_CONFIG_PATH)
    colmap_database_path: Optional[str] = None
    colmap_input_mode: str = "crop"


# ---------------------------------------------------------------------------
# Tracking result
# ---------------------------------------------------------------------------


@dataclass
class VGGTTrackingResult:
    """Container for the optional VGGT tracking pipeline outputs.

    Attributes:
        tracks: Array shaped ``(num_frames, num_tracks, 2)`` giving per-frame pixel coordinates.
        visibilities: Array shaped ``(num_frames, num_tracks)`` with per-frame visibility scores.
        confidences: Optional array containing per-track confidence values (may be ``None``).
        points_3d: Optional array of per-track 3D points (may be ``None``).
        colors: Optional array of per-track RGB colors in ``uint8`` range ``[0, 255]`` (may be ``None``).
    """

    tracks: np.ndarray
    visibilities: np.ndarray
    confidences: Optional[np.ndarray]
    points_3d: Optional[np.ndarray]
    colors: Optional[np.ndarray]


# ---------------------------------------------------------------------------
# Track selection helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TrackSelectionCandidate:
    """Track candidate ranked by geometric support and reprojection error."""

    track_id: int
    track_length: int
    mean_reprojection_error: float
    patches_by_image: dict[int, tuple[int, int]]


def _compute_image_patch(
    u: float,
    v: float,
    image_width: float,
    image_height: float,
    patch_size: int,
) -> tuple[int, int]:
    """Map a 2D pixel measurement to an image patch in an ``NxN`` grid."""
    if patch_size <= 0:
        return (0, 0)

    col = int(np.clip(u, 0.0, image_width) // patch_size)
    row = int(np.clip(v, 0.0, image_height) // patch_size)
    return (row, col)


def _select_track_ids_for_ba_coverage(
    candidates: Sequence[_TrackSelectionCandidate],
    min_track_length: int,
    max_reproj_error: float,
) -> set[int]:
    """Select tracks to maximize patch coverage using ranking by track geometry quality."""
    if not candidates:
        return set()

    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (-candidate.track_length, candidate.mean_reprojection_error, candidate.track_id),
    )

    selected_track_ids: set[int] = set()
    covered_patches: set[tuple[int, int, int]] = set()
    for candidate_track in ordered_candidates:
        if (
            candidate_track.track_length < min_track_length
            or candidate_track.mean_reprojection_error > max_reproj_error
        ):
            continue
        candidate_patch_keys = {
            (image_idx, patch_id[0], patch_id[1]) for image_idx, patch_id in candidate_track.patches_by_image.items()
        }
        if not candidate_patch_keys:
            continue
        if candidate_patch_keys.issubset(covered_patches):
            continue
        selected_track_ids.add(candidate_track.track_id)
        covered_patches.update(candidate_patch_keys)

    return selected_track_ids


def _is_point_in_front_of_camera(camera, point_xyz: np.ndarray, *, epsilon: float = 1e-6) -> bool:
    """Return True if ``point_xyz`` projects in front of ``camera``."""
    if camera is None:
        return False
    try:
        x, y, z = (float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]))
        cam_point = camera.pose().transformTo(Point3(x, y, z))
    except Exception:
        return False
    z_val = cam_point[2] if isinstance(cam_point, np.ndarray) else cam_point.z()
    return float(z_val) > epsilon


# ---------------------------------------------------------------------------
# VGGT tracking internals
# ---------------------------------------------------------------------------


def _import_vggsfm_utils():
    """Return the vendored vggsfm utilities module from the VGGT submodule."""
    from gtsfm.frontend.vggt_geometry_transformer import _USING_FASTVGGT, _import_from_vanilla_vggt

    try:
        from vggt.dependency import vggsfm_utils as _vggsfm_utils  # type: ignore
    except ImportError as exc:
        if _USING_FASTVGGT:
            try:
                tracker_module = _import_from_vanilla_vggt("dependency.vggsfm_utils")
                logger.info("Using vggsfm utilities from the vanilla VGGT submodule.")
                return tracker_module
            except ImportError as fallback_exc:
                exc = fallback_exc
        hint = (
            "Could not import VGGT tracker utilities. Ensure the 'vggt' submodule is checked out by "
            "running `git submodule update --init --recursive`."
        )
        if _USING_FASTVGGT:
            hint += " FastVGGT does not bundle the tracker code, so the vanilla VGGT submodule must remain accessible."
        raise ImportError(hint) from exc
    return _vggsfm_utils


def generate_rank_by_dino(
    images, query_frame_num, image_size=336, model_name="dinov2_vitb14_reg", device="cuda", spatial_similarity=False
):
    """Generate a ranking of frames using DINO ViT features.

    Args:
        images: Tensor of shape (S, 3, H, W) with values in range [0, 1]
        query_frame_num: Number of frames to select
        image_size: Size to resize images to before processing
        model_name: Name of the DINO model to use
        device: Device to run the model on
        spatial_similarity: Whether to use spatial token similarity or CLS token similarity

    Returns:
        List of frame indices ranked by their representativeness
    """
    vggsfm_utils = _import_vggsfm_utils()

    images = F.interpolate(images, (image_size, image_size), mode="bilinear", align_corners=False)

    device_str = str(device)
    cache_key = (model_name, device_str)
    if cache_key in _DINO_V2_MODEL_CACHE:
        dino_v2_model = _DINO_V2_MODEL_CACHE[cache_key]
    else:
        logger.info("⏳ Loading DINOv2 model (%s)...", model_name)
        dino_v2_model = torch.hub.load("facebookresearch/dinov2", model_name)
        dino_v2_model.eval()
        dino_v2_model = dino_v2_model.to(device)
        _DINO_V2_MODEL_CACHE[cache_key] = dino_v2_model
        logger.info("✅ DINOv2 model loaded successfully.")

    imagenet_mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    images_imagenet_norm = (images - imagenet_mean) / imagenet_std

    with torch.no_grad():
        frame_feat = dino_v2_model(images_imagenet_norm, is_training=True)

    if spatial_similarity:
        frame_feat = frame_feat["x_norm_patchtokens"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)
        frame_feat_norm = frame_feat_norm.permute(1, 0, 2)
        similarity_matrix = torch.bmm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))
        similarity_matrix = similarity_matrix.mean(dim=0)
    else:
        frame_feat = frame_feat["x_norm_clstoken"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)
        similarity_matrix = torch.mm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))

    distance_matrix = 100 - similarity_matrix.clone()
    similarity_matrix.fill_diagonal_(-100)
    similarity_sum = similarity_matrix.sum(dim=1)
    most_common_frame_index = torch.argmax(similarity_sum).item()
    fps_idx = vggsfm_utils.farthest_point_sampling(distance_matrix, query_frame_num, most_common_frame_index)

    del frame_feat, frame_feat_norm, similarity_matrix, distance_matrix
    return fps_idx


def _predict_vggt_tracks(model: Any, images: torch.Tensor, query_points: torch.Tensor):
    """Run the VGGT track head, releasing backbone tokens before allocating correlations.

    Call under the same no-grad/autocast context as the model's standard track head.
    All queries are still processed jointly, preserving the tracker's spatial attention.
    """
    aggregated_tokens_list, ps_idx = model.aggregator(images)
    feature_maps = model.track_head.feature_extractor(aggregated_tokens_list, images, ps_idx)
    del aggregated_tokens_list
    if images.is_cuda:
        # Correlation construction and normalization briefly need two full volumes. Release
        # unused backbone allocations only under memory pressure; larger GPUs keep their cache.
        b, frames, _, height, width = feature_maps.shape
        correlation_bytes = b * frames * query_points.shape[1] * height * width * feature_maps.element_size()
        free_bytes, _ = torch.cuda.mem_get_info(images.device)
        if free_bytes < 2 * correlation_bytes:
            torch.cuda.empty_cache()
    return model.track_head.tracker(query_points=query_points, fmaps=feature_maps, iters=model.track_head.iters)


def _run_vggt_head_tracking(
    geo_output: GeometryTransformerOutput,
    *,
    model: Any,
    config: TrackingConfig,
) -> VGGTTrackingResult:
    """Generate dense feature tracks using the VGGT track head."""
    vggsfm_utils = _import_vggsfm_utils()

    device = geo_output.device
    if device.type != "cuda":
        raise RuntimeError(
            "VGGT tracking requires a CUDA-capable GPU because DINO relies on flash attention. "
            "Re-run the pipeline with CUDA available."
        )

    images = geo_output.images
    if images.device != device or images.dtype != torch.float32:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)

    frame_num = images.shape[0]
    if config.use_all_frames_forward_only:
        query_frame_indexes = list(range(frame_num))
    else:
        query_frame_indexes = generate_rank_by_dino(
            images,
            query_frame_num=config.query_frame_num,
            image_size=518,
            model_name="dinov2_vitb14_reg",
            device=device,
            spatial_similarity=False,
        )
        if 0 in query_frame_indexes:
            query_frame_indexes.remove(0)
        query_frame_indexes = [0, *query_frame_indexes]

    extractors = vggsfm_utils.initialize_feature_extractors(
        max_query_num=config.max_query_pts,
        extractor_method=config.keypoint_extractor,
        device=device,
    )

    dense_points = geo_output.dense_points
    depth_confidence = geo_output.depth_confidence
    height, width = images.shape[-2:]

    pred_tracks = []
    pred_vis_scores = []
    pred_conf_scores = []
    pred_world_points = []
    pred_world_points_conf = []
    pred_colors = []

    for query_index in query_frame_indexes:
        query_image = images[query_index]
        query_points = vggsfm_utils.extract_keypoints(query_image, extractors, round_keypoints=True)
        if query_points is None or query_points.shape[1] == 0:
            continue

        query_points = query_points[:, torch.randperm(query_points.shape[1], device=device)]
        if query_points.shape[1] > config.max_query_pts:
            query_points = query_points[:, : config.max_query_pts]

        query_points_round = query_points.squeeze(0).round().long()
        query_points_round[:, 0] = query_points_round[:, 0].clamp(0, width - 1)
        query_points_round[:, 1] = query_points_round[:, 1].clamp(0, height - 1)

        pred_color = (
            images[query_index][:, query_points_round[:, 1], query_points_round[:, 0]].permute(1, 0).cpu().numpy()
            * 255.0
        ).astype(np.uint8)

        pred_point_3d = dense_points[query_index][query_points_round[:, 1], query_points_round[:, 0]]

        pred_conf = None
        if depth_confidence is not None:
            pred_conf = depth_confidence[query_index][query_points_round[:, 1], query_points_round[:, 0]]

        if query_points.shape[1] == 0:
            continue

        reorder_index = vggsfm_utils.calculate_index_mappings(query_index, frame_num, device=device)
        reorder_images = vggsfm_utils.switch_tensor_order([images], reorder_index, dim=0)[0]

        if device.type == "cuda":
            autocast_ctx: Any = amp_autocast("cuda", dtype=geo_output.dtype)
        else:
            autocast_ctx = nullcontext()

        with torch.no_grad():
            with autocast_ctx:
                track_list, vis_scores, conf_scores = _predict_vggt_tracks(model, reorder_images[None], query_points)

        pred_track = track_list[-1]
        pred_track = pred_track.squeeze(0)
        vis_scores = vis_scores.squeeze(0)
        conf_scores = conf_scores.squeeze(0)
        reordered = vggsfm_utils.switch_tensor_order([pred_track, vis_scores, conf_scores], reorder_index, dim=0)
        pred_track, pred_vis, pred_conf_score = reordered

        if config.use_all_frames_forward_only:
            valid_frames_mask = torch.arange(frame_num, device=device) >= query_index
            pred_vis = pred_vis * valid_frames_mask[:, None].to(dtype=pred_vis.dtype)
            if pred_conf_score is not None:
                pred_conf_score = pred_conf_score * valid_frames_mask[:, None].to(dtype=pred_conf_score.dtype)

        pred_tracks.append(pred_track)
        pred_vis_scores.append(pred_vis)
        if pred_conf_score is not None:
            pred_conf_scores.append(pred_conf_score)
        pred_world_points.append(pred_point_3d)
        if pred_conf is not None:
            pred_world_points_conf.append(pred_conf)
        pred_colors.append(pred_color)

    if not pred_tracks:
        empty_tracks = np.zeros((frame_num, 0, 2), dtype=np.float32)
        empty_vis = np.zeros((frame_num, 0), dtype=np.float32)
        empty_conf = np.zeros((0,), dtype=np.float32) if depth_confidence is not None else None
        empty_points = np.zeros((0, 3), dtype=np.float32)
        empty_colors = np.zeros((0, 3), dtype=np.uint8)
        return VGGTTrackingResult(
            tracks=empty_tracks,
            visibilities=empty_vis,
            confidences=empty_conf,
            points_3d=empty_points,
            colors=empty_colors,
        )

    tracks = torch.cat(pred_tracks, dim=1)
    vis_scores = torch.cat(pred_vis_scores, dim=1)
    confidences = torch.cat(pred_conf_scores, dim=1) if pred_conf_scores else None
    points_3d = torch.cat(pred_world_points, dim=0) if pred_world_points else None
    points_3d_conf = torch.cat(pred_world_points_conf, dim=0) if pred_world_points_conf else None
    colors = np.concatenate(pred_colors, axis=0) if pred_colors else None

    if points_3d_conf is not None and points_3d is not None:
        filtered_flag = points_3d_conf > 1.5
        if int(filtered_flag.sum().item()) > config.max_query_pts // 2:
            tracks = tracks[:, filtered_flag]
            vis_scores = vis_scores[:, filtered_flag]
            if confidences is not None:
                confidences = confidences[:, filtered_flag]
            points_3d = points_3d[filtered_flag]
            points_3d_conf = points_3d_conf[filtered_flag]
            if colors is not None:
                colors = colors[filtered_flag.cpu().numpy()]

    return VGGTTrackingResult(
        tracks=tracks.float().cpu().numpy(),
        visibilities=vis_scores.float().cpu().numpy(),
        confidences=confidences.float().cpu().numpy() if confidences is not None else None,
        points_3d=points_3d.float().cpu().numpy() if points_3d is not None else None,
        colors=colors,
    )


# ---------------------------------------------------------------------------
# MultiViewTracker
# ---------------------------------------------------------------------------


class MultiViewTracker:
    """Produces feature tracks and assembles GtsfmData from geometry transformer output."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()

    def run_tracking(
        self,
        geo_output: GeometryTransformerOutput,
        *,
        model: Any = None,
        config: TrackingConfig | None = None,
        image_names: Sequence[str] | None = None,
        dataset_dir: PathLike | None = None,
    ) -> VGGTTrackingResult:
        """Generate dense feature tracks using the VGGT tracking backend.

        Args:
            geo_output: Output from a geometry transformer.
            model: Loaded VGGT model (required for track head).
            config: Optional override config.
            image_names: Optional image names to keep the interface aligned with other trackers.
            dataset_dir: Optional dataset directory from the loader.

        Returns:
            :class:`VGGTTrackingResult` with tracks, visibilities, and 3D points.
        """
        cfg = config or self.config
        if model is None:
            raise ValueError("VGGT tracking requires a loaded VGGT model.")
        return _run_vggt_head_tracking(geo_output, model=model, config=cfg)

    def build_gtsfm_data(
        self,
        geo_output: GeometryTransformerOutput,
        original_coords: torch.Tensor,
        *,
        image_indices: Sequence[int],
        image_names: Optional[Sequence[str]] = None,
        tracking_result: VGGTTrackingResult | None = None,
        points_3d: np.ndarray | None = None,
        points_rgb: np.ndarray | None = None,
        config: TrackingConfig | None = None,
        cluster_label: Optional[str] = None,
    ) -> GtsfmData:
        """Convert geometry transformer output + optional tracks into GtsfmData.

        This creates cameras from extrinsics/intrinsics and optionally inserts
        tracked feature observations as SfmTracks.

        Args:
            geo_output: Output from a geometry transformer.
            original_coords: Tensor shaped ``(N, 6)`` with crop/pad metadata.
            image_indices: Global image indices for each frame.
            image_names: Optional image filenames.
            tracking_result: Optional tracking result to convert into tracks.
            points_3d: Optional dense point cloud (used for track colors).
            points_rgb: Optional per-point RGB colors.
            config: Optional override config.
            cluster_label: Optional cluster name for log messages.

        Returns:
            :class:`GtsfmData` with cameras and optionally tracks.
        """
        cfg = config or self.config

        extrinsic_np = geo_output.extrinsic.to(torch.float32).cpu().numpy()
        intrinsic_np = geo_output.intrinsic.to(torch.float32).cpu().numpy()
        original_coords_np = original_coords.to(torch.float32).cpu().numpy()
        image_names_str = [str(name) for name in image_names] if image_names is not None else None

        gtsfm_data = GtsfmData(number_images=len(image_indices))

        for local_idx, global_idx in enumerate(image_indices):
            image_width = float(original_coords_np[local_idx, 4])
            image_height = float(original_coords_np[local_idx, 5])
            scaled_intrinsic = intrinsic_np[local_idx]

            camera = torch_utils.camera_from_matrices(
                extrinsic_np[local_idx],
                scaled_intrinsic,
                crop_coords=original_coords_np[local_idx],
                use_cal3_bundler=not cfg.ba_use_undistorted_camera_model,
            )
            gtsfm_data.add_camera(global_idx, camera)  # type: ignore[arg-type]
            gtsfm_data.set_image_info(
                global_idx,
                name=image_names_str[local_idx] if image_names_str is not None else None,
                shape=(int(image_height), int(image_width)),
            )

        if tracking_result:
            self._insert_tracks(
                gtsfm_data,
                tracking_result,
                original_coords_np=original_coords_np,
                image_indices=image_indices,
                points_rgb=points_rgb,
                config=cfg,
                cluster_label=cluster_label,
            )

        return gtsfm_data

    def _insert_tracks(
        self,
        gtsfm_data: GtsfmData,
        tracking_result: VGGTTrackingResult,
        *,
        original_coords_np: np.ndarray,
        image_indices: Sequence[int],
        points_rgb: np.ndarray | None,
        config: TrackingConfig,
        cluster_label: Optional[str] = None,
    ) -> None:
        """Insert tracked features as SfmTracks into gtsfm_data (in-place)."""
        import gtsfm.utils.reprojection as reprojection_utils
        from gtsfm.common.sfm_track import SfmMeasurement

        track_mask = tracking_result.visibilities > config.track_vis_thresh

        if tracking_result.confidences is not None and tracking_result.confidences.size > 0:
            confidence_threshold = min(
                config.track_conf_thresh,
                float(np.mean(tracking_result.confidences) - np.std(tracking_result.confidences)),
            )
            track_mask = np.logical_and(track_mask, tracking_result.confidences > confidence_threshold)

        inlier_num = track_mask.sum(0)
        valid_mask = inlier_num >= config.min_track_length
        valid_idx = np.nonzero(valid_mask)[0]
        cameras = gtsfm_data.cameras()

        candidate_tracks: list[tuple[int, Any]] = []
        selection_candidates: list[_TrackSelectionCandidate] = []

        for valid_id in valid_idx:
            rgb: np.ndarray
            if tracking_result.colors is not None and valid_id < tracking_result.colors.shape[0]:
                rgb = tracking_result.colors[valid_id]
            elif points_rgb is not None and valid_id < points_rgb.shape[0]:
                rgb = points_rgb[valid_id]
            else:
                rgb = np.zeros(3, dtype=np.uint8)
            point_xyz = tracking_result.points_3d[valid_id]
            per_track_measurements: list[tuple[int, float, float]] = []
            patches_by_image: dict[int, tuple[int, int]] = {}
            frame_idx = np.where(track_mask[:, valid_id])[0]
            for local_id in frame_idx:
                global_idx = image_indices[local_id]
                u, v = tracking_result.tracks[local_id, valid_id]
                u = u + original_coords_np[local_id, 0]
                v = v + original_coords_np[local_id, 1]

                camera = gtsfm_data.get_camera(global_idx)
                if not _is_point_in_front_of_camera(camera, point_xyz):
                    continue
                per_track_measurements.append((global_idx, u, v))
                patches_by_image[global_idx] = _compute_image_patch(
                    u=u,
                    v=v,
                    image_width=original_coords_np[local_id, 4],
                    image_height=original_coords_np[local_id, 5],
                    patch_size=config.ba_track_patch_size,
                )

            if len(per_track_measurements) < config.min_track_length:
                continue

            sfm_measurements = [
                SfmMeasurement(i=global_idx, uv=np.array([float_u, float_v], dtype=np.float64))
                for global_idx, float_u, float_v in per_track_measurements
            ]
            reproj_errors, _ = reprojection_utils.compute_point_reprojection_errors(
                cameras, point_xyz, sfm_measurements
            )
            valid_reproj_errors = reproj_errors[~np.isnan(reproj_errors)]
            if valid_reproj_errors.size < config.min_track_length:
                continue
            track_length = int(valid_reproj_errors.size)
            mean_reprojection_error = float(np.mean(valid_reproj_errors))

            track = torch_utils.colored_track_from_point(point_xyz, rgb)
            for global_idx, float_u, float_v in per_track_measurements:
                track.addMeasurement(global_idx, Point2(float_u, float_v))
            min_triangulation_angle = config.min_triangulation_angle
            if min_triangulation_angle > 0.0:
                import gtsfm.utils.tracks as track_utils

                if track_utils.get_max_triangulation_angle(track, cameras) < min_triangulation_angle:
                    continue

            candidate_tracks.append((valid_id, track))
            selection_candidates.append(
                _TrackSelectionCandidate(
                    track_id=valid_id,
                    track_length=track_length,
                    mean_reprojection_error=mean_reprojection_error,
                    patches_by_image=patches_by_image,
                )
            )

        if config.enable_ba_track_patching:
            selected_track_ids = _select_track_ids_for_ba_coverage(
                selection_candidates, config.min_track_length, config.vggt_max_reproj_error
            )
        else:
            selected_track_ids = {track_id for track_id, _ in candidate_tracks}
        for track_id, track in candidate_tracks:
            if track_id in selected_track_ids:
                gtsfm_data.add_track(track)

        logger.info(
            "num valid tracks after filtering%s: %d out of %d",
            " and patching" if config.enable_ba_track_patching else " (patching disabled)",
            gtsfm_data.number_tracks(),
            len(valid_idx),
        )


# ---------------------------------------------------------------------------
# ColmapTracker
# ---------------------------------------------------------------------------


class ColmapTracker(MultiViewTracker):
    """Tracker implementation that reads matches from a cached COLMAP database."""

    _config_cache: dict[Path, dict[str, Any]] = {}
    _database_cache: dict[Path, _CachedColmapDatabase] = {}
    _image_size_cache: dict[Path, tuple[int, int]] = {}

    def __init__(self, config: TrackingConfig | None = None) -> None:
        super().__init__(config=config)

    def run_tracking(
        self,
        geo_output: GeometryTransformerOutput,
        *,
        model: Any = None,
        config: TrackingConfig | None = None,
        image_names: Sequence[str] | None = None,
        dataset_dir: PathLike | None = None,
    ) -> VGGTTrackingResult:
        """Load tracks from a COLMAP database and package as tracking output.

        Args:
            geo_output: Geometry transformer output containing dense tracks and images.
            model: Unused for this tracker.
            config: Optional override config.
            image_names: Image names used to resolve COLMAP image IDs.
            dataset_dir: Dataset directory from the active loader.
        """
        cfg = config or self.config
        if model is not None:
            logger.debug("Ignoring provided model argument for ColmapTracker.")

        if not image_names:
            logger.warning(
                "ColmapTracker requires image names to resolve COLMAP keypoints; returning empty tracks."
            )
            frame_num = int(geo_output.images.shape[0])
            return VGGTTrackingResult(
                tracks=np.zeros((frame_num, 0, 2), dtype=np.float32),
                visibilities=np.zeros((frame_num, 0), dtype=np.float32),
                confidences=None,
                points_3d=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
            )

        db_path = self._resolve_database_path(cfg, dataset_dir=dataset_dir)
        if not Path(db_path).exists():
            raise FileNotFoundError(f"COLMAP database not found: {db_path}")

        db_data = self._load_database(db_path)

        selected_image_ids = self._resolve_image_ids(db_data, image_names or [])
        if not selected_image_ids:
            logger.warning("ColmapTracker received no matching image names for database %s.", db_path)
            frame_num = int(geo_output.images.shape[0])
            return VGGTTrackingResult(
                tracks=np.zeros((frame_num, 0, 2), dtype=np.float32),
                visibilities=np.zeros((frame_num, 0), dtype=np.float32),
                confidences=None,
                points_3d=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
            )

        if len(selected_image_ids) != len(image_names or []):
            logger.warning(
                "ColmapTracker mapped %d of %d provided image names to database images.",
                len(selected_image_ids),
                len(image_names or []),
            )

        preprocessed_tracks, visibilities = self._load_tracks_for_images(
            db_data=db_data,
            selected_image_ids=selected_image_ids,
            mode=cfg.colmap_input_mode,
        )

        num_tracks = preprocessed_tracks.shape[1]
        if num_tracks == 0:
            empty_points = np.zeros((0, 3), dtype=np.float32)
            return VGGTTrackingResult(
                tracks=preprocessed_tracks,
                visibilities=visibilities,
                confidences=None,
                points_3d=empty_points,
                colors=np.zeros((0, 3), dtype=np.uint8),
            )

        dense_points = geo_output.dense_points
        images = geo_output.images
        points_3d = self._sample_points_from_tracks(preprocessed_tracks, visibilities, dense_points)
        colors = self._sample_colors_from_tracks(preprocessed_tracks, visibilities, images)
        return VGGTTrackingResult(
            tracks=preprocessed_tracks,
            visibilities=visibilities,
            confidences=None,
            points_3d=points_3d,
            colors=colors,
        )

    @classmethod
    def _load_reference_config(cls, config_path: Path) -> dict[str, Any]:
        if config_path in cls._config_cache:
            return cls._config_cache[config_path]

        if not config_path.exists():
            logger.debug("COLMAP tracker config path not found: %s", config_path)
            cls._config_cache[config_path] = {}
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            logger.warning("Invalid YAML content in %s; expected a mapping.", config_path)
            loaded = {}
        cls._config_cache[config_path] = loaded
        return loaded

    @classmethod
    def _resolve_database_path(
        cls,
        config: TrackingConfig,
        *,
        dataset_dir: PathLike | None = None,
    ) -> Path:
        def _resolve_relative_path(path_value: Path) -> Path:
            if path_value.is_absolute():
                return path_value
            # Runtime CLI paths are typically relative to the process CWD (repo root).
            cwd_candidate = (Path.cwd() / path_value).resolve()
            if cwd_candidate.exists():
                return cwd_candidate
            # Backward-compatible fallback for configs historically interpreted relative to package root.
            return (Path(__file__).resolve().parent.parent / path_value).resolve()

        def _resolve_db_from_dataset_dir(dataset_dir_value: PathLike) -> Path | None:
            dataset_path = _resolve_relative_path(Path(dataset_dir_value).expanduser())
            db_candidate = dataset_path / "database.db"
            if db_candidate.exists():
                return db_candidate
            return None

        if config.colmap_database_path:
            explicit = Path(config.colmap_database_path).expanduser()
            return _resolve_relative_path(explicit)

        if dataset_dir:
            db_candidate = _resolve_db_from_dataset_dir(dataset_dir)
            if db_candidate is not None:
                return db_candidate

        reference_cfg = cls._load_reference_config(Path(config.colmaptracker_config_path).expanduser())
        loader_cfg = reference_cfg.get("loader", {})
        dataset_dir = loader_cfg.get("dataset_dir")
        if isinstance(dataset_dir, str) and dataset_dir not in {"???", ""}:
            db_candidate = _resolve_db_from_dataset_dir(dataset_dir)
            if db_candidate is not None:
                return db_candidate

        raise FileNotFoundError(
            "Could not resolve a COLMAP database path. "
            "Set 'colmap_database_path' explicitly in the tracker config or ensure "
            "'loader.dataset_dir' in the colmaptracker config points to a directory "
            "containing 'database.db'."
        )

    @classmethod
    def _load_database(cls, db_path: Path) -> _CachedColmapDatabase:
        db_path = db_path.resolve()
        if db_path in cls._database_cache:
            return cls._database_cache[db_path]

        connection = sqlite3.connect(str(db_path))
        cursor = connection.cursor()

        image_rows = cursor.execute("SELECT image_id, name FROM images").fetchall()
        image_id_by_name: dict[int, str] = {}
        name_to_image_id: dict[str, int] = {}
        for image_id, raw_name in image_rows:
            image_id_int = int(image_id)
            normalized = cls._normalize_image_name(str(raw_name))
            image_id_by_name[image_id_int] = str(raw_name)
            name_to_image_id[normalized] = image_id_int
            name_to_image_id[os.path.basename(normalized)] = image_id_int

        keypoints_by_image_id: dict[int, np.ndarray] = {}
        for image_id, rows, cols, data in cursor.execute("SELECT image_id, rows, cols, data FROM keypoints"):
            image_id_int = int(image_id)
            if rows == 0 or cols == 0 or data is None:
                keypoints_by_image_id[image_id_int] = np.zeros((0, 2), dtype=np.float32)
                continue
            blob = np.frombuffer(data, dtype=np.float32).reshape(int(rows), int(cols))
            keypoints_by_image_id[image_id_int] = blob[:, :2].copy()

        matches: list[tuple[int, int, np.ndarray]] = []
        for pair_id, rows, cols, data in cursor.execute("SELECT pair_id, rows, cols, data FROM matches"):
            if rows == 0 or cols == 0 or data is None:
                continue
            image_id1, image_id2 = cls._pair_id_to_image_ids(int(pair_id))
            blob = np.frombuffer(data, dtype=np.uint32).reshape(int(rows), int(cols))
            matches.append((image_id1, image_id2, blob[:, :2].copy()))

        connection.close()

        cached = _CachedColmapDatabase(
            db_path=db_path,
            image_id_by_name=image_id_by_name,
            name_to_image_id=name_to_image_id,
            keypoints_by_image_id=keypoints_by_image_id,
            matches=matches,
        )
        cls._database_cache[db_path] = cached
        return cached

    @classmethod
    def _pair_id_to_image_ids(cls, pair_id: int) -> tuple[int, int]:
        image_id2 = pair_id % _COLMAP_MAX_IMAGE_ID
        image_id1 = (pair_id - image_id2) // _COLMAP_MAX_IMAGE_ID
        return int(image_id1), int(image_id2)

    @classmethod
    def _normalize_image_name(cls, name: str) -> str:
        return os.path.normpath(name.replace("\\", "/"))

    @classmethod
    def _resolve_image_ids(cls, db_data: _CachedColmapDatabase, image_names: Sequence[str]) -> list[int]:
        image_ids: list[int] = []
        for raw_name in image_names:
            if raw_name is None:
                continue
            normalized = cls._normalize_image_name(str(raw_name))
            image_id = db_data.name_to_image_id.get(normalized)
            if image_id is None:
                image_id = db_data.name_to_image_id.get(os.path.basename(normalized))
            if image_id is None:
                logger.warning("Image %s not found in COLMAP database %s.", raw_name, db_data.db_path)
                continue
            image_ids.append(image_id)
        seen: set[int] = set()
        deduped = [image_id for image_id in image_ids if not (image_id in seen or seen.add(image_id))]
        return deduped

    def _resolve_image_size(self, db_dir: Path, image_name: str) -> tuple[int, int]:
        normalized = self._normalize_image_name(image_name)
        candidates = [
            db_dir / normalized,
            db_dir / os.path.basename(normalized),
            Path(normalized),
            Path(os.path.expanduser(normalized)),
        ]
        for candidate in candidates:
            candidate = candidate.expanduser()
            if not candidate.is_absolute():
                candidate = (db_dir / candidate).resolve()
            cached_size = self._image_size_cache.get(candidate)
            if cached_size is not None:
                return cached_size
            if candidate.exists():
                with PILImage.open(candidate) as img:
                    size = img.size
                self._image_size_cache[candidate] = size
                return size
        return 0, 0

    def _load_tracks_for_images(
        self,
        db_data: _CachedColmapDatabase,
        selected_image_ids: list[int],
        mode: str = "crop",
    ) -> tuple[np.ndarray, np.ndarray]:
        selected_set = set(selected_image_ids)
        num_frames = len(selected_image_ids)

        image_id_to_local_frame = {image_id: frame_idx for frame_idx, image_id in enumerate(selected_image_ids)}
        image_name_by_id = {image_id: db_data.image_id_by_name.get(image_id, "") for image_id in selected_image_ids}

        # Read image sizes and map DB keypoints to the preprocessing coordinate frame.
        db_dir = db_data.db_path.parent
        frame_points: list[np.ndarray] = []
        frame_sizes: list[tuple[int, int]] = []
        for image_id in selected_image_ids:
            keypoints = db_data.keypoints_by_image_id.get(image_id, np.zeros((0, 2), dtype=np.float32))
            if keypoints.size == 0:
                frame_points.append(np.zeros((0, 2), dtype=np.float32))
                frame_sizes.append((0, 0))
                continue
            width, height = self._resolve_image_size(db_dir, image_name_by_id[image_id])
            if width == 0 or height == 0:
                width = float(np.max(keypoints[:, 0]) + 1.0) if keypoints.size else 1.0
                height = float(np.max(keypoints[:, 1]) + 1.0) if keypoints.size else 1.0
                width = max(1.0, width)
                height = max(1.0, height)
            transformed = self._transform_keypoints_to_preprocessed_frame(
                keypoints.astype(np.float32),
                int(width),
                int(height),
                mode=mode,
            )
            frame_points.append(transformed.points)
            frame_sizes.append((transformed.height, transformed.width))

        # Additional global padding to emulate batched preprocessing in the loader.
        max_h = max(h for h, _ in frame_sizes)
        max_w = max(w for _, w in frame_sizes)
        if max_h == 0:
            max_h = 1
        if max_w == 0:
            max_w = 1
        for idx, (frame_h, frame_w) in enumerate(frame_sizes):
            h_pad = max_h - frame_h
            w_pad = max_w - frame_w
            if h_pad <= 0 and w_pad <= 0:
                continue
            x_offset = w_pad // 2
            y_offset = h_pad // 2
            if frame_points[idx].size:
                frame_points[idx][:, 0] = frame_points[idx][:, 0] + x_offset
                frame_points[idx][:, 1] = frame_points[idx][:, 1] + y_offset
            frame_points[idx] = np.clip(frame_points[idx], [0.0, 0.0], [max_w - 1.0, max_h - 1.0])

        # Union-Find over keypoint observations for selected image pair matches.
        num_local_points = []
        for points in frame_points:
            num_local_points.append(points.shape[0])
        total_keypoints = int(sum(num_local_points))
        if total_keypoints == 0:
            return (
                np.zeros((num_frames, 0, 2), dtype=np.float32),
                np.zeros((num_frames, 0), dtype=np.float32),
            )

        local_offset: dict[int, int] = {}
        running = 0
        for image_id, num_points in zip(selected_image_ids, num_local_points):
            local_offset[image_id] = running
            running += num_points

        parent = np.arange(total_keypoints, dtype=np.int64)
        rank = np.zeros(total_keypoints, dtype=np.int8)

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return int(idx)

        def union(i_idx: int, j_idx: int) -> None:
            i_root = find(i_idx)
            j_root = find(j_idx)
            if i_root == j_root:
                return
            if rank[i_root] < rank[j_root]:
                parent[i_root] = j_root
            elif rank[i_root] > rank[j_root]:
                parent[j_root] = i_root
            else:
                parent[j_root] = i_root
                rank[i_root] += 1

        for image_id1, image_id2, pair_matches in db_data.matches:
            if image_id1 not in selected_set or image_id2 not in selected_set:
                continue
            if pair_matches.size == 0:
                continue
            offset1 = local_offset[image_id1]
            offset2 = local_offset[image_id2]
            num_p1 = num_local_points[image_id_to_local_frame[image_id1]]
            num_p2 = num_local_points[image_id_to_local_frame[image_id2]]
            match_kp1 = pair_matches[:, 0].astype(np.int64)
            match_kp2 = pair_matches[:, 1].astype(np.int64)
            valid = (match_kp1 >= 0) & (match_kp1 < num_p1) & (match_kp2 >= 0) & (match_kp2 < num_p2)
            if not np.any(valid):
                continue
            match_kp1 = match_kp1[valid]
            match_kp2 = match_kp2[valid]
            for i1, i2 in zip(offset1 + match_kp1, offset2 + match_kp2):
                union(int(i1), int(i2))

        roots = np.fromiter((find(idx) for idx in range(total_keypoints)), dtype=np.int64, count=total_keypoints)
        track_obs: dict[int, list[int]] = {}
        for idx, root in enumerate(roots):
            track_obs.setdefault(int(root), []).append(idx)
        valid_track_obs = [obs for obs in track_obs.values() if len(obs) >= 2]
        num_tracks = len(valid_track_obs)
        if num_tracks == 0:
            return (
                np.zeros((num_frames, 0, 2), dtype=np.float32),
                np.zeros((num_frames, 0), dtype=np.float32),
            )

        tracks = np.zeros((num_frames, num_tracks, 2), dtype=np.float32)
        visibility = np.zeros((num_frames, num_tracks), dtype=np.float32)
        frame_index_by_global = np.empty(total_keypoints, dtype=np.int32)
        keypoint_index_by_global = np.empty(total_keypoints, dtype=np.int32)
        for image_id, frame_idx in image_id_to_local_frame.items():
            start = local_offset[image_id]
            end = start + num_local_points[frame_idx]
            frame_index_by_global[start:end] = frame_idx
            keypoint_index_by_global[start:end] = np.arange(num_local_points[frame_idx], dtype=np.int32)

        for track_idx, obs in enumerate(valid_track_obs):
            for local_idx in obs:
                frame_idx = int(frame_index_by_global[local_idx])
                kp_idx = int(keypoint_index_by_global[local_idx])
                points = frame_points[frame_idx]
                if kp_idx < points.shape[0]:
                    tracks[frame_idx, track_idx, :] = points[kp_idx]
                    visibility[frame_idx, track_idx] = 1.0

        return tracks, visibility

    @classmethod
    def _transform_keypoints_to_preprocessed_frame(
        cls,
        keypoints: np.ndarray,
        orig_width: int,
        orig_height: int,
        mode: str = "crop",
    ) -> "_TransformedKeypoints":
        target_size = _COLMAP_TRACKER_TARGET_SIZE
        if keypoints.size == 0 or orig_width <= 0 or orig_height <= 0:
            return _TransformedKeypoints(np.zeros((0, 2), dtype=np.float32), 0, 0)
        if mode not in {"crop", "pad"}:
            raise ValueError(f"Unsupported input mode '{mode}'. Expected 'crop' or 'pad'.")

        orig_w = float(orig_width)
        orig_h = float(orig_height)
        scaled_w = float(target_size)
        scaled_h = round(orig_h * (scaled_w / orig_w) / 14.0) * 14.0
        scale_x = scaled_w / orig_w
        scale_y = scaled_h / orig_h
        x = keypoints[:, 0] * scale_x
        y = keypoints[:, 1] * scale_y

        if mode == "pad":
            h_padding = target_size - int(scaled_h)
            w_padding = target_size - int(scaled_w)
            pad_left = max(w_padding // 2, 0)
            pad_top = max(h_padding // 2, 0)
            x = x + float(pad_left)
            y = y + float(pad_top)
            out_h = target_size
            out_w = target_size
        else:
            start_y = (int(scaled_h) - target_size) // 2 if int(scaled_h) > target_size else 0
            y = y - float(start_y)
            out_h = min(int(scaled_h), target_size)
            out_w = min(int(scaled_w), target_size)

        x = np.clip(x, 0.0, max(float(out_w - 1), 0.0))
        y = np.clip(y, 0.0, max(float(out_h - 1), 0.0))
        return _TransformedKeypoints(np.stack([x, y], axis=1).astype(np.float32), out_h, out_w)

    @staticmethod
    def _sample_points_from_tracks(
        tracks: np.ndarray,
        visibilities: np.ndarray,
        dense_points: torch.Tensor,
    ) -> np.ndarray:
        if tracks.size == 0 or visibilities.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        dense_np = dense_points.detach().float().cpu().numpy()
        visibility_mask = visibilities > 0.0
        first_frame = visibility_mask.argmax(axis=0)
        valid = visibility_mask.sum(axis=0) > 0
        track_indices = np.flatnonzero(valid)
        if track_indices.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        h = dense_np.shape[1]
        w = dense_np.shape[2]
        sample_frames = first_frame[track_indices]
        sample_x = tracks[sample_frames, track_indices, 0]
        sample_y = tracks[sample_frames, track_indices, 1]
        sample_x_int = np.rint(sample_x).astype(np.int64)
        sample_y_int = np.rint(sample_y).astype(np.int64)
        sample_x_int = np.clip(sample_x_int, 0, w - 1)
        sample_y_int = np.clip(sample_y_int, 0, h - 1)
        sampled = dense_np[sample_frames.astype(np.int64), sample_y_int, sample_x_int]
        sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
        points = np.zeros((tracks.shape[1], 3), dtype=np.float32)
        points[track_indices] = sampled.astype(np.float32)
        return points

    @staticmethod
    def _sample_colors_from_tracks(
        tracks: np.ndarray,
        visibilities: np.ndarray,
        images: torch.Tensor,
    ) -> np.ndarray:
        if tracks.size == 0 or visibilities.size == 0:
            return np.zeros((0, 3), dtype=np.uint8)

        image_np = (images.detach().float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        visibility_mask = visibilities > 0.0
        first_frame = visibility_mask.argmax(axis=0)
        valid = visibility_mask.sum(axis=0) > 0
        track_indices = np.flatnonzero(valid)
        if track_indices.size == 0:
            return np.zeros((0, 3), dtype=np.uint8)

        h = image_np.shape[2]
        w = image_np.shape[3]
        sample_frames = first_frame[track_indices]
        sample_x = tracks[sample_frames, track_indices, 0]
        sample_y = tracks[sample_frames, track_indices, 1]
        sample_x_int = np.rint(sample_x).astype(np.int64)
        sample_y_int = np.rint(sample_y).astype(np.int64)
        sample_x_int = np.clip(sample_x_int, 0, w - 1)
        sample_y_int = np.clip(sample_y_int, 0, h - 1)
        sampled = image_np[sample_frames.astype(np.int64), :, sample_y_int, sample_x_int].transpose(0, 1)
        colors = sampled.astype(np.uint8)
        out = np.zeros((tracks.shape[1], 3), dtype=np.uint8)
        out[track_indices] = colors
        return out


@dataclass
class _TransformedKeypoints:
    points: np.ndarray
    height: int
    width: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ColmapTracker",
    "MultiViewTracker",
    "TrackingConfig",
    "VGGTTrackingResult",
    "generate_rank_by_dino",
]
