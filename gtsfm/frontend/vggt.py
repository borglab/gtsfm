"""Helpers to integrate VGGT with GTSFM without relying on pycolmap."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from gtsam import Point2  # type: ignore
from torch.amp import autocast as amp_autocast  # type: ignore

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils import logger as logger_utils
from gtsfm.utils import torch as torch_utils

PathLike = Union[str, Path]

logger = logger_utils.get_logger()

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRDPARTY_ROOT = REPO_ROOT / "thirdparty"
VGGT_SUBMODULE_PATH = THIRDPARTY_ROOT / "vggt"
LIGHTGLUE_SUBMODULE_PATH = THIRDPARTY_ROOT / "LightGlue"
DEFAULT_WEIGHTS_PATH = VGGT_SUBMODULE_PATH / "weights" / "model.pt"


def _ensure_submodule_on_path(path: Path, name: str) -> None:
    """Add a vendored thirdparty module to ``sys.path`` if needed."""
    if not path.exists():
        raise ImportError(
            f"Required submodule '{name}' not found at {path}. "
            "Run 'git submodule update --init --recursive' to fetch it."
        )

    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_ensure_submodule_on_path(VGGT_SUBMODULE_PATH, "vggt")
_ensure_submodule_on_path(LIGHTGLUE_SUBMODULE_PATH, "LightGlue")

from vggt.models.vggt import VGGT  # type: ignore
from vggt.utils.geometry import unproject_depth_map_to_point_map  # type: ignore
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues  # type: ignore
from vggt.utils.load_fn import load_and_preprocess_images_square  # type: ignore
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore


@dataclass
class VGGTReconstructionConfig:
    """Configuration for the high-level VGGT reconstruction pipeline."""

    vggt_fixed_resolution: int = 518
    img_load_resolution: int = 1024
    max_query_pts: int = 1000
    query_frame_num: int = 4
    fine_tracking: bool = True
    vis_thresh: float = 0.2
    max_reproj_error: float = 8.0
    confidence_threshold: float = 5.0
    max_num_points: int = 100000
    shared_camera: bool = False
    keypoint_extractor: str = "aliked+sp"
    seed: int = 42


@dataclass
class VGGTReconstructionResult:
    """Outputs from the VGGT reconstruction helper."""

    gtsfm_data: GtsfmData
    points_3d: np.ndarray
    points_rgb: Optional[np.ndarray]


def default_dtype(device: torch.device) -> torch.dtype:
    """Pick a floating-point dtype suitable for VGGT on the provided device."""
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device=device)
        return torch.bfloat16 if capability[0] >= 8 else torch.float16
    elif device.type == "mps":
        # MPS has some limitations with float16, so use float32 for now
        # Users can explicitly pass dtype=torch.float16 if they want to try it
        return torch.float32
    else:
        # CPU fallback
        return torch.float32


def resolve_weights_path(weights_path: PathLike | None = None) -> Path:
    """Return a concrete path to the VGGT checkpoint, validating that it exists."""
    checkpoint = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"VGGT checkpoint not found at {checkpoint}. " "Download weights via `scripts/download_model_weights.sh`."
        )
    return checkpoint


def load_model(
    weights_path: PathLike | None = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> VGGT:
    """Load the VGGT model weights on the requested device."""
    resolved_device = torch_utils.default_device(device)
    checkpoint = resolve_weights_path(weights_path)

    model = VGGT()
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(resolved_device)

    # Move to the desired dtype if provided (mainly for FP16/BF16 GPUs).
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


def _rescale_intrinsic_for_original_resolution(
    intrinsic: np.ndarray,
    reconstruction_resolution: int,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Adapt intrinsics estimated on a square crop back to the original image size."""
    resized = intrinsic.copy()
    resize_ratio = max(image_width, image_height) / float(reconstruction_resolution)
    resized[:2, :] *= resize_ratio
    resized[0, 2] = image_width / 2.0
    resized[1, 2] = image_height / 2.0
    return resized


def _convert_measurement_to_original_resolution(
    uv_inference: Tuple[float, float],
    original_coord: np.ndarray,
    inference_resolution: int,
    img_load_resolution: int,
) -> Tuple[float, float]:
    """Convert VGGT inference coordinates back to the original image coordinate system."""

    x_infer, y_infer = uv_inference
    x1, y1 = original_coord[0], original_coord[1]
    width, height = original_coord[4], original_coord[5]

    # VGGT runs on the ``img_load_resolution`` square; inference down-samples that square to the
    # (typically smaller) ``inference_resolution``. Undo that downscale so we can use the crop
    # metadata stored in ``original_coord``.
    scale_back_to_load = float(img_load_resolution) / float(inference_resolution)
    x_load = x_infer * scale_back_to_load
    y_load = y_infer * scale_back_to_load

    # ``original_coord`` encodes the location of the original, possibly rectangular, image within
    # the padded square (in *load* resolution). Remove the padding and scale the remaining pixels
    # back to the native resolution.
    max_side = float(max(width, height))
    resize_ratio = max_side / float(img_load_resolution)
    u = (x_load - x1) * resize_ratio
    v = (y_load - y1) * resize_ratio
    u = float(np.clip(u, 0.0, max(width - 1, 0)))
    v = float(np.clip(v, 0.0, max(height - 1, 0)))
    return u, v


def _unproject_to_colored_points(
    *,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth_map: np.ndarray,
    depth_confidence: np.ndarray,
    image_batch: torch.Tensor,
    config: VGGTReconstructionConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert raw VGGT predictions into point attributes."""
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    inference_resolution = config.vggt_fixed_resolution
    points_rgb_tensor = F.interpolate(
        image_batch.detach().cpu(),
        size=(inference_resolution, inference_resolution),
        mode="bilinear",
        align_corners=False,
    )
    points_rgb = (points_rgb_tensor.to(torch.float32).numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    points_xyf = create_pixel_coordinate_grid(points_3d.shape[0], points_3d.shape[1], points_3d.shape[2])

    conf_mask = depth_confidence >= config.confidence_threshold
    conf_mask = randomly_limit_trues(conf_mask, config.max_num_points)
    return points_3d[conf_mask], points_rgb[conf_mask], points_xyf[conf_mask]


def _convert_vggt_outputs_to_gtsfm_data(
    *,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    original_coords: torch.Tensor,
    image_indices: Sequence[int],
    image_names: Optional[Sequence[str]],
    config: VGGTReconstructionConfig,
    points_3d: np.ndarray,
    points_rgb: np.ndarray,
    points_xyf: np.ndarray,
) -> GtsfmData:
    """Convert raw VGGT predictions into ``GtsfmData``."""

    original_coords_np = original_coords.detach().cpu().numpy()
    image_names_str = [str(name) for name in image_names] if image_names is not None else None

    gtsfm_data = GtsfmData(number_images=len(image_indices))

    for local_idx, global_idx in enumerate(image_indices):
        image_width = float(original_coords_np[local_idx, 4])
        image_height = float(original_coords_np[local_idx, 5])

        scaled_intrinsic = _rescale_intrinsic_for_original_resolution(
            intrinsic[local_idx], config.vggt_fixed_resolution, image_width, image_height
        )
        camera = torch_utils.camera_from_matrices(extrinsic[local_idx], scaled_intrinsic)
        gtsfm_data.add_camera(global_idx, camera)  # type: ignore[arg-type]
        gtsfm_data.set_image_info(
            global_idx,
            name=image_names_str[local_idx] if image_names_str is not None else None,
            shape=(int(image_height), int(image_width)),
        )

    if points_3d.size > 0 and points_rgb is not None:
        for j, xyz in enumerate(points_3d):
            xyf = points_xyf[j]
            frame_float = float(xyf[2])
            frame_idx = int(np.clip(round(frame_float), 0, len(image_indices) - 1))
            u, v = _convert_measurement_to_original_resolution(
                (float(xyf[0]), float(xyf[1])),
                original_coords_np[frame_idx],
                config.vggt_fixed_resolution,
                config.img_load_resolution,
            )
            track = torch_utils.colored_track_from_point(xyz, points_rgb[j])
            global_idx = image_indices[frame_idx]
            track.addMeasurement(global_idx, Point2(u, v))
            gtsfm_data.add_track(track)

    expected_indices = set(int(i) for i in image_indices)
    valid_camera_indices = set(gtsfm_data.get_valid_camera_indices())
    if valid_camera_indices != expected_indices:
        logger.warning(
            "VGGT cluster returned cameras with indices %s, expected %s.",
            sorted(valid_camera_indices),
            sorted(expected_indices),
        )

    for j, track in enumerate(gtsfm_data.get_tracks()):
        for meas_idx in range(track.numberMeasurements()):
            cam_idx, _ = track.measurement(meas_idx)
            if cam_idx not in expected_indices:
                logger.warning(
                    "VGGT track %d references camera %d not in cluster indices %s.",
                    j,
                    cam_idx,
                    sorted(expected_indices),
                )
                break

    return gtsfm_data


def run_reconstruction(
    image_batch: torch.Tensor,
    *,
    image_indices: Sequence[int],
    image_names: Optional[Sequence[str]] = None,
    original_coords: torch.Tensor,
    config: Optional[VGGTReconstructionConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    model: Optional[VGGT] = None,
    weights_path: PathLike | None = None,
) -> VGGTReconstructionResult:
    """Run VGGT on a batch of images and convert outputs to ``GtsfmData``."""
    cfg = config or VGGTReconstructionConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    resolved_device = torch_utils.default_device(device)
    resolved_dtype = dtype or default_dtype(resolved_device)

    if model is None:
        model = load_model(weights_path, device=resolved_device, dtype=resolved_dtype)
    else:
        model = model.to(resolved_device)
        if resolved_dtype is not None:
            model = model.to(dtype=resolved_dtype)  # type: ignore
        model.eval()  # type: ignore

    # Ensure tensors match the model's device and dtype
    image_batch = image_batch.to(resolved_device, dtype=resolved_dtype)
    original_coords = original_coords.to(resolved_device)

    inference_resolution = cfg.vggt_fixed_resolution
    images_for_model = F.interpolate(
        image_batch,
        size=(inference_resolution, inference_resolution),
        mode="bilinear",
        align_corners=False,
    )

    if resolved_device.type == "cuda":
        autocast_ctx: Any = amp_autocast("cuda", dtype=resolved_dtype)
    elif resolved_device.type == "mps":
        # MPS doesn't support autocast with custom dtype, so we use nullcontext
        # but the model will still run on MPS with the dtype we set earlier
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = nullcontext()

    with torch.no_grad():
        with autocast_ctx:
            assert model is not None  # for mypy
            batched = images_for_model.unsqueeze(0)
            tokens, ps_idx = model.aggregator(batched)
            pose_enc = model.camera_head(tokens)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batched.shape[-2:])
            depth_map, depth_conf = model.depth_head(tokens, batched, ps_idx)

    extrinsic_np = extrinsic.squeeze(0).cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).cpu().numpy()
    depth_map_np = depth_map.squeeze(0).cpu().numpy()
    depth_conf_np = depth_conf.squeeze(0).cpu().numpy()

    if depth_conf_np.ndim == 4 and depth_conf_np.shape[-1] == 1:
        depth_conf_np = np.squeeze(depth_conf_np, axis=-1)

    points_3d, points_rgb, points_xyf = _unproject_to_colored_points(
        config=cfg,
        image_batch=image_batch,
        extrinsic=extrinsic_np,
        intrinsic=intrinsic_np,
        depth_map=depth_map_np,
        depth_confidence=depth_conf_np,
    )

    gtsfm_data = _convert_vggt_outputs_to_gtsfm_data(
        config=cfg,
        extrinsic=extrinsic_np,
        intrinsic=intrinsic_np,
        original_coords=original_coords,
        image_indices=image_indices,
        image_names=image_names,
        points_3d=points_3d,
        points_rgb=points_rgb,
        points_xyf=points_xyf,
    )

    return VGGTReconstructionResult(
        gtsfm_data=gtsfm_data,
        points_3d=points_3d,
        points_rgb=points_rgb,
    )


def run_VGGT(model: VGGT, images: torch.Tensor, dtype: torch.dtype, resolution: int = 518):
    """Keep a thin wrapper around the original VGGT demo helper for compatibility."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("VGGT expects images shaped as (N, 3, H, W).")

    # Determine the device type from the model
    model_device = next(model.parameters()).device

    # Ensure the images match the model's device and dtype
    images = images.to(model_device, dtype=dtype)
    resized = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    if model_device.type == "cuda":
        autocast_ctx: Any = amp_autocast("cuda", dtype=dtype)
    else:
        # For MPS and CPU, we don't use autocast with custom dtype
        autocast_ctx = nullcontext()

    with torch.no_grad():
        with autocast_ctx:
            batched = resized.unsqueeze(0)
            tokens, ps_idx = model.aggregator(batched)
            pose_enc = model.camera_head(tokens)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batched.shape[-2:])
            depth_map, depth_conf = model.depth_head(tokens, batched, ps_idx)

    return (
        extrinsic.squeeze(0).cpu().numpy(),
        intrinsic.squeeze(0).cpu().numpy(),
        depth_map.squeeze(0).cpu().numpy(),
        depth_conf.squeeze(0).cpu().numpy(),
    )


# --- vggt_tracking -------------------------------------------------


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


def _import_predict_tracks():
    """Return the vendored ``predict_tracks`` helper from the VGGT submodule.

    The tracker lives in ``thirdparty/vggt``. We keep this import behind a small helper so that runtime
    errors surface with a clear explanation when the submodule is missing.
    """

    try:
        from vggt.dependency.track_predict import predict_tracks as _predict_tracks  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when the submodule is absent
        raise ImportError(
            "Could not import VGGT tracker utilities. Ensure the 'vggt' submodule is checked out by "
            "running `git submodule update --init --recursive`."
        ) from exc
    return _predict_tracks


def run_vggt_tracking(
    images: torch.Tensor,
    *,
    depth_confidence: Optional[Union[np.ndarray, torch.Tensor]] = None,
    dense_points_3d: Optional[Union[np.ndarray, torch.Tensor]] = None,
    config: Optional[VGGTReconstructionConfig] = None,
    tracker_kwargs: Optional[dict[str, Any]] = None,
) -> VGGTTrackingResult:
    """Generate dense feature tracks using the VGGSfM tracker shipped with VGGT.

    Parameters:
        images: Tensor shaped ``(num_frames, 3, H, W)`` at the *square* VGGT load resolution. You can reuse
            the ``images`` tensor that you passed into :func:`run_reconstruction`; typically this is the output
            from ``load_and_preprocess_images_square`` prior to interpolation.
        depth_confidence: Optional dense confidence volume produced by VGGT's depth head. When using the value
            returned from :func:`run_reconstruction`, pass the ``depth_conf_np`` array you stored before calling
            :func:`_unproject_to_colored_points`. The tensor is expected to align with ``images`` spatially, i.e.
            be square with shape ``(num_frames, 1, H, W)``.
        dense_points_3d: Optional dense point cloud in VGGT's inference frame. Provide the per-pixel 3D points
            prior to the outlier pruning applied in :func:`_unproject_to_colored_points` so that the tracker can
            look up metric depth at query locations. The expected shape is ``(num_frames, H, W, 3)``.
        config: Optional :class:`VGGTReconstructionConfig`. We reuse the existing configuration container because
            it already captures the tracker-specific parameters (``max_query_pts``, ``query_frame_num``, etc.).
        tracker_kwargs: Optional dictionary to override individual keyword arguments passed to the underlying
            :func:`vggt.dependency.track_predict.predict_tracks` function. This is useful if you want to tweak
            settings not exposed via :class:`VGGTReconstructionConfig`.

    Returns:
        :class:`VGGTTrackingResult` aggregating the numpy arrays emitted by the tracker. The visibility scores can
        be thresholded manually, e.g. ``mask = result.visibilities > config.vis_thresh``. The tracks are expressed
        in the same *square* coordinate frame as ``images``; remember to rescale them back to the original image
        crop using :func:`_convert_measurement_to_original_resolution` if you plan to add them to ``GtsfmData``.

    Example:
        >>> extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, image_batch, dtype)
        >>> dense_points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        >>> cfg = VGGTReconstructionConfig()
        >>> tracking = run_vggt_tracking(
        ...     images=image_batch,
        ...     depth_confidence=depth_conf,
        ...     dense_points_3d=dense_points,
        ...     config=cfg,
        ... )
        >>> high_quality = tracking.visibilities > cfg.vis_thresh
        >>> first_track_pixels = tracking.tracks[:, 0]
    """

    cfg = config or VGGTReconstructionConfig()
    predict_tracks = _import_predict_tracks()

    device = images.device
    dtype = images.dtype

    def _to_matching_tensor(value: Optional[Union[np.ndarray, torch.Tensor]]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        return torch.from_numpy(value).to(device=device, dtype=dtype)

    conf_tensor = _to_matching_tensor(depth_confidence)
    points_tensor = _to_matching_tensor(dense_points_3d)

    tracker_args: dict[str, Any] = {
        "max_query_pts": cfg.max_query_pts,
        "query_frame_num": cfg.query_frame_num,
        "keypoint_extractor": cfg.keypoint_extractor,
        "fine_tracking": cfg.fine_tracking,
    }
    if tracker_kwargs:
        tracker_args.update(tracker_kwargs)

    tracks, vis_scores, confidences, points_3d, colors = predict_tracks(
        images,
        conf=conf_tensor,
        points_3d=points_tensor,
        masks=None,
        **tracker_args,
    )

    return VGGTTrackingResult(
        tracks=tracks,
        visibilities=vis_scores,
        confidences=confidences,
        points_3d=points_3d,
        colors=colors,
    )


__all__ = [
    "VGGTReconstructionConfig",
    "VGGTReconstructionResult",
    "DEFAULT_WEIGHTS_PATH",
    "VGGT_SUBMODULE_PATH",
    "LIGHTGLUE_SUBMODULE_PATH",
    "default_dtype",
    "load_and_preprocess_images_square",
    "resolve_weights_path",
    "load_model",
    "run_VGGT",
    "run_reconstruction",
    "_import_predict_tracks",
    "run_vggt_tracking",
    "VGGTTrackingResult",
]
