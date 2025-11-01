"""Helpers to integrate VGGT with GTSFM without relying on pycolmap."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import gtsam  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from gtsam import Point2, Point3, Pose3, Rot3, SfmTrack  # type: ignore
from torch.amp import autocast as amp_autocast  # type: ignore

import gtsfm.common.types as gtsfm_types
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils import logger as logger_utils

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
    max_points_for_colmap: int = 100000
    shared_camera: bool = False
    keypoint_extractor: str = "aliked+sp"
    seed: int = 42


@dataclass
class VGGTReconstructionResult:
    """Outputs from the VGGT reconstruction helper."""

    gtsfm_data: GtsfmData
    reconstruction_resolution: int
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    depth_map: np.ndarray
    depth_confidence: np.ndarray
    points_3d: np.ndarray
    points_rgb: Optional[np.ndarray]
    points_xyf: Optional[np.ndarray]
    image_indices: Tuple[int, ...]
    used_ba: bool
    valid_track_mask: Optional[np.ndarray]
    fallback_reason: Optional[str] = None


def default_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve a concrete device for VGGT inference."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    """Pick a floating-point dtype suitable for VGGT on the provided device."""
    if device.type != "cuda":
        return torch.float32
    capability = torch.cuda.get_device_capability(device=device)
    return torch.bfloat16 if capability[0] >= 8 else torch.float16


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
    resolved_device = default_device(device)
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


def _pose_from_extrinsic(matrix: np.ndarray) -> Pose3:
    """Convert a VGGT extrinsic matrix (camera-from-world) to a Pose3 (world-from-camera)."""
    cRw: np.ndarray = matrix[:3, :3]
    t = matrix[:3, 3]
    return Pose3(Rot3(cRw), Point3(*t)).inverse()


def _calibration_from_intrinsic(matrix: np.ndarray) -> gtsam.Cal3_S2:
    """Map a 3x3 intrinsic matrix to a Cal3_S2."""
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    cx = float(matrix[0, 2])
    cy = float(matrix[1, 2])
    return gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)


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
    total_num_images: Optional[int] = None,
) -> VGGTReconstructionResult:
    """Run VGGT on a batch of images and convert outputs to ``GtsfmData``."""
    cfg = config or VGGTReconstructionConfig()
    total_num_images = total_num_images or (max(image_indices) + 1)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    resolved_device = default_device(device)
    resolved_dtype = dtype or default_dtype(resolved_device)

    if model is None:
        model = load_model(weights_path, device=resolved_device, dtype=resolved_dtype)
    else:
        model = model.to(resolved_device)
        model.eval()  # type: ignore

    image_batch = image_batch.to(resolved_device)
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

    points_3d = unproject_depth_map_to_point_map(depth_map_np, extrinsic_np, intrinsic_np)
    points_rgb_tensor = F.interpolate(
        image_batch.cpu(),
        size=(inference_resolution, inference_resolution),
        mode="bilinear",
        align_corners=False,
    )
    points_rgb = (points_rgb_tensor.numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    points_xyf = create_pixel_coordinate_grid(points_3d.shape[0], points_3d.shape[1], points_3d.shape[2])

    conf_mask = depth_conf_np >= cfg.confidence_threshold
    conf_mask = randomly_limit_trues(conf_mask, cfg.max_points_for_colmap)

    points_3d_flat = points_3d[conf_mask]
    points_rgb_flat = points_rgb[conf_mask] if points_rgb is not None else None
    points_xyf_flat = points_xyf[conf_mask]

    original_coords_np = original_coords.detach().cpu().numpy()
    image_names_str = [str(name) for name in image_names] if image_names is not None else None

    gtsfm_data = GtsfmData(number_images=total_num_images)

    for local_idx, global_idx in enumerate(image_indices):
        pose = _pose_from_extrinsic(extrinsic_np[local_idx])
        image_width = float(original_coords_np[local_idx, 4])
        image_height = float(original_coords_np[local_idx, 5])

        scaled_intrinsic = _rescale_intrinsic_for_original_resolution(
            intrinsic_np[local_idx], inference_resolution, image_width, image_height
        )
        calibration = _calibration_from_intrinsic(scaled_intrinsic)
        camera_cls = gtsfm_types.get_camera_class_for_calibration(calibration)
        gtsfm_data.add_camera(global_idx, camera_cls(pose, calibration))  # type: ignore[arg-type]
        gtsfm_data.set_image_info(
            global_idx,
            name=image_names_str[local_idx] if image_names_str is not None else None,
            shape=(int(image_height), int(image_width)),
        )

    if points_3d_flat.size > 0 and points_rgb_flat is not None:
        for idx, xyz in enumerate(points_3d_flat):
            xyf = points_xyf_flat[idx]
            frame_float = float(xyf[2])
            frame_idx = int(np.clip(round(frame_float), 0, len(image_indices) - 1))
            u, v = _convert_measurement_to_original_resolution(
                (float(xyf[0]), float(xyf[1])),
                original_coords_np[frame_idx],
                inference_resolution,
                cfg.img_load_resolution,
            )
            track = SfmTrack(Point3(*xyz))
            color = points_rgb_flat[idx]
            track.r = float(color[0])
            track.g = float(color[1])
            track.b = float(color[2])
            global_idx = image_indices[frame_idx]
            track.addMeasurement(global_idx, Point2(u, v))
            gtsfm_data.add_track(track)

    fallback_reason = None
    if points_3d_flat.size == 0:
        fallback_reason = "VGGT produced no confident depth values; reconstruction contains cameras only."

    valid_camera_indices = set(gtsfm_data.get_valid_camera_indices())
    expected_indices = set(int(idx) for idx in image_indices)
    if valid_camera_indices != expected_indices:
        logger.warning(
            "VGGT cluster returned cameras with indices %s, expected %s.",
            sorted(valid_camera_indices),
            sorted(expected_indices),
        )

    for track_idx, track in enumerate(gtsfm_data.get_tracks()):
        for meas_idx in range(track.numberMeasurements()):
            cam_idx, _ = track.measurement(meas_idx)
            if cam_idx not in expected_indices:
                logger.warning(
                    "VGGT track %d references camera %d not in cluster indices %s.",
                    track_idx,
                    cam_idx,
                    sorted(expected_indices),
                )
                break

    return VGGTReconstructionResult(
        gtsfm_data=gtsfm_data,
        reconstruction_resolution=inference_resolution,
        extrinsic=extrinsic_np,
        intrinsic=intrinsic_np,
        depth_map=depth_map_np,
        depth_confidence=depth_conf_np,
        points_3d=points_3d_flat,
        points_rgb=points_rgb_flat,
        points_xyf=points_xyf_flat,
        image_indices=tuple(image_indices),
        used_ba=False,
        valid_track_mask=None,
        fallback_reason=fallback_reason,
    )


def run_VGGT(model: VGGT, images: torch.Tensor, dtype: torch.dtype, resolution: int = 518):
    """Keep a thin wrapper around the original VGGT demo helper for compatibility."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("VGGT expects images shaped as (N, 3, H, W).")
    resized = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    if torch.cuda.is_available():
        autocast_ctx: Any = amp_autocast("cuda", dtype=dtype)
    else:
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


__all__ = [
    "VGGTReconstructionConfig",
    "VGGTReconstructionResult",
    "DEFAULT_WEIGHTS_PATH",
    "VGGT_SUBMODULE_PATH",
    "LIGHTGLUE_SUBMODULE_PATH",
    "default_device",
    "default_dtype",
    "load_and_preprocess_images_square",
    "resolve_weights_path",
    "load_model",
    "run_VGGT",
    "run_reconstruction",
]
