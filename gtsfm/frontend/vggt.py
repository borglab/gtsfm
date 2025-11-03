"""Helpers to integrate VGGT with GTSFM without relying on pycolmap."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast as amp_autocast  # type: ignore

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils import logger as logger_utils
from gtsfm.utils import torch as torch_utils

PathLike = Union[str, Path]

logger = logger_utils.get_logger()

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRDPARTY_ROOT = REPO_ROOT / "thirdparty"
VGGT_SUBMODULE_PATH = THIRDPARTY_ROOT / "vggt"
FASTVGGT_SUBMODULE_PATH = THIRDPARTY_ROOT / "FastVGGT"
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


_USING_FASTVGGT = False
if FASTVGGT_SUBMODULE_PATH.exists():
    try:
        _ensure_submodule_on_path(FASTVGGT_SUBMODULE_PATH, "FastVGGT")
        _USING_FASTVGGT = True
    except ImportError:
        _USING_FASTVGGT = False

_ensure_submodule_on_path(VGGT_SUBMODULE_PATH, "vggt")
if _USING_FASTVGGT:
    fast_path = str(FASTVGGT_SUBMODULE_PATH)
    if fast_path in sys.path:
        sys.path.remove(fast_path)
    sys.path.insert(0, fast_path)
_ensure_submodule_on_path(LIGHTGLUE_SUBMODULE_PATH, "LightGlue")

from vggt.models.vggt import VGGT  # type: ignore

if _USING_FASTVGGT:
    logger.info("⚡ FastVGGT enabled via thirdparty/FastVGGT.")
else:
    logger.info("📷 Using vanilla VGGT (FastVGGT submodule not detected).")
from vggt.utils.helper import randomly_limit_trues  # type: ignore
from vggt.utils.load_fn import load_and_preprocess_images_square  # type: ignore
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

from gtsfm.frontend.anysplat import (
    batchify_unproject_depth_map_to_point_map as _anysplat_batchify_unproject,
)  # type: ignore

DEFAULT_FIXED_RESOLUTION = 518

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _resolve_dtype_argument(arg: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
    """Convert a config-friendly dtype specifier into a ``torch.dtype``."""
    if arg is None:
        return None
    if isinstance(arg, torch.dtype):
        return arg
    if isinstance(arg, str):
        key = arg.lower()
        if key in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[key]
        candidate = getattr(torch, key, None)
        if isinstance(candidate, torch.dtype):
            return candidate
        raise ValueError(f"Unrecognized torch dtype string '{arg}'.")
    raise TypeError(f"Unsupported dtype specifier of type {type(arg)!r}: {arg!r}")


@dataclass
class VggtConfiguration:
    """Configuration for the high-level VGGT reconstruction pipeline."""

    img_load_resolution: int = 1024
    vggt_fixed_resolution: int = DEFAULT_FIXED_RESOLUTION
    seed: int = 42
    confidence_threshold: float = 5.0
    max_num_points: int = 100000
    dtype: Optional[Union[str, torch.dtype]] = None
    model_ctor_kwargs: dict[str, Any] = field(default_factory=dict)
    use_sparse_attention: bool = False

    # Tracking-specific parameters:
    tracking: bool = False
    max_query_pts: int = 1000
    query_frame_num: int = 4
    keypoint_extractor: str = "aliked+sp"
    fine_tracking: bool = True
    vis_thresh: float = 0.2
    max_reproj_error: float = 8.0  # TODO(Frank): Does not seem to be used


@dataclass
class VggtOutput:
    """Outputs produced by a single VGGT forward pass."""

    device: torch.device
    dtype: torch.dtype
    resized_images: torch.Tensor
    extrinsic: torch.Tensor
    intrinsic: torch.Tensor
    depth_map: torch.Tensor
    depth_confidence: torch.Tensor
    dense_points: torch.Tensor


@dataclass
class VggtReconstruction:
    """Outputs from the VGGT reconstruction helper."""

    gtsfm_data: GtsfmData
    points_3d: np.ndarray
    points_rgb: np.ndarray


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
    model_kwargs: Optional[dict[str, Any]] = None,
) -> VGGT:
    """Load the VGGT model weights on the requested device."""
    resolved_device = torch_utils.default_device(device)
    checkpoint = resolve_weights_path(weights_path)

    ctor_kwargs = dict(model_kwargs) if model_kwargs else {}
    try:
        model = VGGT(**ctor_kwargs)
    except TypeError as exc:
        hint = "Ensure your thirdparty/vggt checkout provides the requested functionality."
        if ctor_kwargs and not _USING_FASTVGGT:
            hint += " (FastVGGT submodule is required for options such as 'merging'.)"
        raise TypeError(f"Failed to construct VGGT with custom arguments {ctor_kwargs}. {hint}") from exc
    state_dict = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        logger.warning("VGGT checkpoint had unexpected keys (ignored): %s", unexpected[:5])
    if missing:
        logger.warning("VGGT checkpoint missing keys (ignored): %s", missing[:5])
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
    """Convert VGGT vggt_output coordinates back to the original image coordinate system."""

    x_infer, y_infer = uv_inference
    x1, y1 = original_coord[0], original_coord[1]
    width, height = original_coord[4], original_coord[5]

    # VGGT runs on the ``img_load_resolution`` square; vggt_output down-samples that square to the
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


def _high_confidence_pointcloud(config: VggtConfiguration, vggt_output: VggtOutput) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw VGGT predictions into point attributes."""
    points_3d = vggt_output.dense_points.to(torch.float32).cpu().numpy()
    points_rgb = (vggt_output.resized_images.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(
        np.uint8
    )

    depth_conf_np = vggt_output.depth_confidence.to(torch.float32).cpu().numpy()
    conf_mask = depth_conf_np >= config.confidence_threshold
    conf_mask = randomly_limit_trues(conf_mask, config.max_num_points)  # limit number of points if asked
    return points_3d[conf_mask], points_rgb[conf_mask]


def _convert_vggt_outputs_to_gtsfm_data(
    *,
    vggt_output: VggtOutput,
    original_coords: torch.Tensor,
    image_indices: Sequence[int],
    image_names: Optional[Sequence[str]],
    config: VggtConfiguration,
    points_3d: np.ndarray,
    points_rgb: np.ndarray,
) -> GtsfmData:
    """Convert raw VGGT predictions into ``GtsfmData``."""

    extrinsic_np = vggt_output.extrinsic.to(torch.float32).cpu().numpy()
    intrinsic_np = vggt_output.intrinsic.to(torch.float32).cpu().numpy()
    original_coords_np = original_coords.to(torch.float32).cpu().numpy()
    image_names_str = [str(name) for name in image_names] if image_names is not None else None

    gtsfm_data = GtsfmData(number_images=len(image_indices))

    for local_idx, global_idx in enumerate(image_indices):
        image_width = float(original_coords_np[local_idx, 4])
        image_height = float(original_coords_np[local_idx, 5])

        scaled_intrinsic = _rescale_intrinsic_for_original_resolution(
            intrinsic_np[local_idx], config.vggt_fixed_resolution, image_width, image_height
        )
        camera = torch_utils.camera_from_matrices(extrinsic_np[local_idx], scaled_intrinsic)
        gtsfm_data.add_camera(global_idx, camera)  # type: ignore[arg-type]
        gtsfm_data.set_image_info(
            global_idx,
            name=image_names_str[local_idx] if image_names_str is not None else None,
            shape=(int(image_height), int(image_width)),
        )

    if points_3d.size > 0 and points_rgb is not None:
        for j, xyz in enumerate(points_3d):
            track = torch_utils.colored_track_from_point(xyz, points_rgb[j])
            gtsfm_data.add_track(track)

    return gtsfm_data


def run_VGGT(
    images: torch.Tensor,
    *,
    config: Optional[VggtConfiguration] = None,
    model: Optional[VGGT] = None,
    weights_path: PathLike | None = None,
) -> VggtOutput:
    """Run VGGT on a batch of images and return raw model predictions.

    Computes the full per-pixel point cloud using the AnySplat acceleration path (when available).
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("VGGT expects images shaped as (N, 3, H, W).")

    resolved_device = torch_utils.default_device()
    cfg = config or VggtConfiguration()
    requested_dtype = _resolve_dtype_argument(cfg.dtype)
    resolved_dtype = requested_dtype or default_dtype(resolved_device)

    config_model_kwargs = dict(cfg.model_ctor_kwargs) if cfg.model_ctor_kwargs else None

    if model is None:
        model = load_model(
            weights_path,
            device=resolved_device,
            dtype=resolved_dtype,
            model_kwargs=config_model_kwargs,
        )
    else:
        model = model.to(resolved_device)
        assert model is not None, "model should not be None here"
        if resolved_dtype is not None:
            model = model.to(dtype=resolved_dtype)
        assert model is not None, "model should not be None here"
        model.eval()

    assert model is not None
    images = images.to(resolved_device, dtype=resolved_dtype)
    res = cfg.vggt_fixed_resolution if cfg else DEFAULT_FIXED_RESOLUTION
    resized_images = F.interpolate(images, size=(res, res), mode="bilinear")

    # FastVGGT requires the model to know the actual patch grid dimensions used for token merging.
    patch_w = max(1, resized_images.shape[-1] // getattr(model.aggregator, "patch_size", 14))
    patch_h = max(1, resized_images.shape[-2] // getattr(model.aggregator, "patch_size", 14))
    if hasattr(model, "update_patch_dimensions"):
        try:
            model.update_patch_dimensions(patch_w, patch_h)
        except Exception as exc:  # pragma: no cover - best effort for FastVGGT compatibility
            logger.warning("Failed to update VGGT patch dimensions (%dx%d): %s", patch_w, patch_h, exc)

    if resolved_device.type == "cuda":
        autocast_ctx: Any = amp_autocast("cuda", dtype=resolved_dtype)
    else:
        autocast_ctx = nullcontext()

    with torch.no_grad():
        with autocast_ctx:
            batched = resized_images.unsqueeze(0)  # make into (training) batch of 1
            tokens, ps_idx = model.aggregator(batched)  # transformer backbone
            pose_enc = model.camera_head(tokens)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batched.shape[-2:])
            depth_map, depth_conf = model.depth_head(tokens, batched, ps_idx)

            assert _anysplat_batchify_unproject is not None, "Anysplat dependencies not available"
            dense_points = _anysplat_batchify_unproject(depth_map, extrinsic, intrinsic)

    depth_confidence = depth_conf.squeeze(0)
    if depth_confidence.ndim == 4 and depth_confidence.shape[-1] == 1:
        depth_confidence = depth_confidence.squeeze(-1)

    return VggtOutput(
        device=resolved_device,
        dtype=resolved_dtype,
        resized_images=resized_images,
        extrinsic=extrinsic.squeeze(0),
        intrinsic=intrinsic.squeeze(0),
        depth_map=depth_map.squeeze(0),
        depth_confidence=depth_confidence,
        dense_points=dense_points.squeeze(0),
    )


# --- VGGT tracking -------------------------------------------------


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
    images: torch.Tensor, vggt_output: VggtOutput, *, config: Optional[VggtConfiguration] = None
) -> VGGTTrackingResult:
    """Generate dense feature tracks using the VGGSfM tracker shipped with VGGT.

    Parameters:
        images: Tensor shaped ``(num_frames, 3, H, W)`` at the *square* VGGT load resolution. You can reuse
            the ``images`` tensor that you passed into :func:`run_reconstruction`; typically this is the output
            from ``load_and_preprocess_images_square`` prior to interpolation.
        vggt_output: Output from :func:`run_VGGT`. The ``depth_confidence`` and optional ``dense_points`` tensors
            are consumed directly, avoiding redundant transfers or recomputation.
        config: Optional :class:`VggtConfiguration`. We reuse the existing configuration container because
            it already captures the tracker-specific parameters (``max_query_pts``, ``query_frame_num``, etc.).
    Returns:
        :class:`VGGTTrackingResult` aggregating the numpy arrays emitted by the tracker. The visibility scores can
        be thresholded manually, e.g. ``mask = result.visibilities > config.vis_thresh``. The tracks are expressed
        in the same *square* coordinate frame as ``images``; remember to rescale them back to the original image
        crop using :func:`_convert_measurement_to_original_resolution` if you plan to add them to ``GtsfmData``.

    Example:
        >>> vggt_output = run_VGGT(image_batch, model=model)
        >>> cfg = VggtConfiguration()
        >>> tracking = run_vggt_tracking(image_batch, vggt_output, config=cfg)
        >>> high_quality = tracking.visibilities > cfg.vis_thresh
        >>> first_track_pixels = tracking.tracks[:, 0]
    """

    cfg = config or VggtConfiguration()
    predict_tracks = _import_predict_tracks()

    # We will track on whatever device the images are already on.
    # TODO(Frank): maybe we should enforce GPU here?
    device = images.device
    dtype = images.dtype
    conf_tensor = vggt_output.depth_confidence.to(device=device, dtype=dtype)
    points_tensor = vggt_output.dense_points.to(device=device, dtype=dtype)

    tracks, vis_scores, confidences, points_3d, colors = predict_tracks(
        images,
        conf=conf_tensor,
        points_3d=points_tensor,
        masks=None,  # ignored anyway !
        max_query_pts=cfg.max_query_pts,
        query_frame_num=cfg.query_frame_num,
        keypoint_extractor=cfg.keypoint_extractor,
        fine_tracking=cfg.fine_tracking,
    )

    return VGGTTrackingResult(
        tracks=tracks, visibilities=vis_scores, confidences=confidences, points_3d=points_3d, colors=colors
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
    images: torch.Tensor, inference: VGGTInferenceResult, *, config: Optional[VGGTReconstructionConfig] = None
) -> VGGTTrackingResult:
    """Generate dense feature tracks using the VGGSfM tracker shipped with VGGT.

    Parameters:
        images: Tensor shaped ``(num_frames, 3, H, W)`` at the *square* VGGT load resolution. You can reuse
            the ``images`` tensor that you passed into :func:`run_reconstruction`; typically this is the output
            from ``load_and_preprocess_images_square`` prior to interpolation.
        vggt_output: Output from :func:`run_VGGT`. The ``depth_confidence`` and optional ``dense_points`` tensors
            are consumed directly, avoiding redundant transfers or recomputation.
        config: Optional :class:`VggtConfiguration`. We reuse the existing configuration container because
            it already captures the tracker-specific parameters (``max_query_pts``, ``query_frame_num``, etc.).
        tracker_kwargs: Optional dictionary to override individual keyword arguments passed to the underlying
            :func:`vggt.dependency.track_predict.predict_tracks` function. This is useful if you want to tweak
            settings not exposed via :class:`VggtConfiguration`.

    Returns:
        :class:`VGGTTrackingResult` aggregating the numpy arrays emitted by the tracker. The visibility scores can
        be thresholded manually, e.g. ``mask = result.visibilities > config.vis_thresh``. The tracks are expressed
        in the same *square* coordinate frame as ``images``; remember to rescale them back to the original image
        crop using :func:`_convert_measurement_to_original_resolution` if you plan to add them to ``GtsfmData``.

    Example:
        >>> vggt_output = run_VGGT(image_batch, model=model, dtype=dtype, return_dense_points=True)
        >>> cfg = VggtConfiguration()
        >>> tracking = run_vggt_tracking(image_batch, vggt_output, config=cfg)
        >>> high_quality = tracking.visibilities > cfg.vis_thresh
        >>> first_track_pixels = tracking.tracks[:, 0]
    """

    cfg = config or VggtConfiguration()
    predict_tracks = _import_predict_tracks()

    # We will track on whatever device the images are already on.
    # TODO(Frank): maybe we should enforce GPU here?
    device = images.device
    dtype = images.dtype
    conf_tensor = vggt_output.depth_confidence.to(device=device, dtype=dtype)
    points_tensor = vggt_output.dense_points.to(device=device, dtype=dtype)

    tracks, vis_scores, confidences, points_3d, colors = predict_tracks(
        images,
        conf=conf_tensor,
        points_3d=points_tensor,
        masks=None,  # ignored anyway !
        max_query_pts=cfg.max_query_pts,
        query_frame_num=cfg.query_frame_num,
        keypoint_extractor=cfg.keypoint_extractor,
        fine_tracking=cfg.fine_tracking,
    )

    return VGGTTrackingResult(
        tracks=tracks, visibilities=vis_scores, confidences=confidences, points_3d=points_3d, colors=colors
    )


# --- VGGT reconstruction -------------------------------------------------


def run_reconstruction(
    images: torch.Tensor,
    *,
    image_indices: Sequence[int],
    image_names: Optional[Sequence[str]] = None,
    original_coords: torch.Tensor,
    config: Optional[VggtConfiguration] = None,
    model: Optional[VGGT] = None,
    weights_path: PathLike | None = None,
) -> VggtReconstruction:
    """Run VGGT on a batch of images and convert outputs to ``GtsfmData``.

    Args:
        images: Tensor shaped ``(num_frames, 3, H, W)`` at the *square* VGGT load resolution. You can
            obtain this tensor by calling ``load_and_preprocess_images_square`` prior to interpolation.
        image_indices: Sequence of global image indices corresponding to the provided ``images`` batch.
        image_names: Optional sequence of image filenames corresponding to the provided ``images`` batch.
        original_coords: Tensor shaped ``(num_frames, 6)`` giving the original image crop metadata
            for each image in ``images``. Each row is ``(x1, y1, x2, y2, width, height)``.
        config: Optional :class:`VggtConfiguration`.
        model: Optional pre-loaded VGGT model. If ``None``, the model is loaded from ``weights_path``.
        weights_path: Optional path to VGGT checkpoint. Ignored if ``model`` is provided.

    Returns:
        :class:`VggtReconstruction` containing the reconstructed ``GtsfmData`` and point cloud.
    """
    cfg = config or VggtConfiguration()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    vggt_output = run_VGGT(images, config=cfg, model=model, weights_path=weights_path)

    if cfg.tracking:
        tracking_result = run_vggt_tracking(images, vggt_output, config=cfg)

    points_3d, points_rgb = _high_confidence_pointcloud(cfg, vggt_output)

    gtsfm_data = _convert_vggt_outputs_to_gtsfm_data(
        config=cfg,
        vggt_output=vggt_output,
        original_coords=original_coords,
        image_indices=image_indices,
        image_names=image_names,
        points_3d=points_3d,
        points_rgb=points_rgb,
    )

    return VggtReconstruction(
        gtsfm_data=gtsfm_data,
        points_3d=points_3d,
        points_rgb=points_rgb,
    )


def run_reconstruction_gtsfm_data_only(images: torch.Tensor, **kwargs) -> GtsfmData:
    """Run VGGT on a batch of images and convert outputs to ``GtsfmData``.

    Args:
        images: Tensor shaped ``(num_frames, 3, H, W)`` at the *square* VGGT load resolution. You can
            obtain this tensor by calling ``load_and_preprocess_images_square`` prior to interpolation.
        **kwargs: Additional keyword arguments passed to :func:`run_reconstruction`.

    Returns:
        The reconstructed ``GtsfmData``.
    """
    result = run_reconstruction(images, **kwargs)
    return result.gtsfm_data


# -------------------------------------------------------------------------

__all__ = [
    "VggtConfiguration",
    "VggtReconstruction",
    "DEFAULT_WEIGHTS_PATH",
    "VGGT_SUBMODULE_PATH",
    "LIGHTGLUE_SUBMODULE_PATH",
    "default_dtype",
    "load_and_preprocess_images_square",
    "resolve_weights_path",
    "load_model",
    "run_VGGT",
    "run_reconstruction",
    "run_vggt_tracking",
    "VGGTTrackingResult",
    "VggtOutput",
]
