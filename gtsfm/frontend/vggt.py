"""Helpers to integrate VGGT with GTSFM without relying on pycolmap."""

from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from gtsam import Point2, Point3
from PIL import Image as PILImage
from torch.amp import autocast as amp_autocast  # type: ignore
from torchvision import transforms as TF

import gtsfm.common.types as gtsfm_types
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils import data_utils
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
_VANILLA_VGGT_NAMESPACE = "_gtsfm_vanilla_vggt"


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


def _import_from_vanilla_vggt(module_suffix: str) -> ModuleType:
    """Import a module from the vanilla VGGT submodule even when FastVGGT shadows ``vggt``."""

    package_root = (VGGT_SUBMODULE_PATH / "vggt").resolve()
    if not package_root.exists():
        raise ImportError(
            f"Vanilla VGGT tracker utilities not found at {package_root}. "
            "Run 'git submodule update --init --recursive' to fetch them."
        )

    alias = _VANILLA_VGGT_NAMESPACE
    namespace = sys.modules.get(alias)
    if namespace is None:
        path_str = str(package_root)
        namespace = ModuleType(alias)
        namespace.__path__ = [path_str]
        namespace.__package__ = alias
        spec = ModuleSpec(alias, loader=None, is_package=True)
        spec.submodule_search_locations = list(namespace.__path__)
        namespace.__spec__ = spec
        sys.modules[alias] = namespace

    full_name = f"{alias}.{module_suffix}"
    return importlib.import_module(full_name)


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
from vggt.utils.geometry import unproject_depth_map_to_point_map  # type: ignore
from vggt.utils.helper import randomly_limit_trues  # type: ignore
from vggt.utils.load_fn import load_and_preprocess_images_square  # type: ignore
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

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


def load_image_batch_vggt_loader(loader, indices: List[int], mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching,
    but VGGT model can also work well with different shapes.

    Args:
        loader: Loader instance providing ``get_image``.
        indices: List of image indices to load.
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                            - "crop" (default): Sets width to 518px and center crops height if needed.
                            - "pad": Preserves all pixels by making the largest dimension 518px
                            and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
        and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
        and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(indices) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for idx in indices:
        # Open image
        img = loader.get_image(idx).value_array

        img = PILImage.fromarray(img)

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), PILImage.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        logger.warning("Found images with different shapes: %s", shapes)
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(indices) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    height, width = images.shape[-2], images.shape[-1]
    coords = np.tile([0.0, 0.0, float(width), float(height), float(width), float(height)], (len(indices), 1))
    original_coords_tensor = torch.from_numpy(coords).float()

    return images, original_coords_tensor


@dataclass
class VggtConfiguration:
    """Configuration for the high-level VGGT reconstruction pipeline."""

    seed: int = 42
    confidence_threshold: float = 5.0
    max_num_points: int = 100000
    dtype: Optional[Union[str, torch.dtype]] = None
    model_ctor_kwargs: dict[str, Any] = field(default_factory=dict)
    use_sparse_attention: bool = False
    run_bundle_adjustment_on_leaf: bool = False
    store_pre_ba_result: bool = False

    # Tracking-specific parameters:
    tracking: bool = True
    max_query_pts: int = 2048
    query_frame_num: int = 3
    track_vis_thresh: float = 0.05
    track_conf_thresh: float = 0.2
    keypoint_extractor: str = "aliked+sp+sift"
    max_reproj_error: float = 8.0
    min_triangulation_angle: float = 0.0


@dataclass
class VggtOutput:  # TODO(Frank): derive from base class shared with AnySplat (in utils.torch.py)
    """Outputs produced by a single VGGT forward pass."""

    device: torch.device
    dtype: torch.dtype
    images: torch.Tensor
    extrinsic: torch.Tensor
    intrinsic: torch.Tensor
    depth_map: torch.Tensor
    depth_confidence: torch.Tensor
    dense_points: torch.Tensor


@dataclass
class VggtReconstruction:
    """Outputs from the VGGT reconstruction helper.

    Attributes:
        gtsfm_data: Sparse scene estimate (post-BA if enabled) in original image coordinates.
        pre_ba_data: Optional sparse scene estimate before bundle adjustment (debug-only).
        points_3d: Dense point cloud filtered by VGGT confidence scores.
        points_rgb: Per-point RGB colors aligned with ``points_3d``.
        tracking_result: Optional dense tracking payload in the square VGGT coordinate frame.
    """

    gtsfm_data: GtsfmData
    points_3d: np.ndarray
    points_rgb: np.ndarray
    pre_ba_data: GtsfmData | None = None
    tracking_result: "VGGTTrackingResult | None" = None

    def visualize_tracks(
        self,
        images: torch.Tensor | np.ndarray,
        *,
        output_dir: PathLike,
        visibility_threshold: float = 0.0,
        frames_per_row: int = 4,
        save_grid: bool = True,
    ) -> None:
        """Overlay tracked feature trajectories on the provided image tensor.

        Args:
            images: Tensor shaped ``(num_frames, 3, H, W)`` (or an array convertible to it) in the same
                coordinate frame used for tracking, i.e. the square VGGT load resolution.
            output_dir: Destination directory for visualization artifacts.
            visibility_threshold: Minimum per-frame visibility score for a track sample to be rendered.
            frames_per_row: Number of frames per row when generating the grid visualization.
            save_grid: Whether to emit the stitched grid image alongside per-frame overlays.
        """
        if self.tracking_result is None:
            raise ValueError("Tracking results are unavailable; ensure VGGT ran with tracking enabled.")

        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(np.asarray(images))

        if images.ndim != 4:
            raise ValueError(
                f"Expected images shaped (num_frames, 3, H, W); received tensor with shape {tuple(images.shape)}."
            )

        images = images.detach().cpu()

        tracks = torch.from_numpy(self.tracking_result.tracks).to(dtype=torch.float32)
        visibilities = torch.from_numpy(self.tracking_result.visibilities).to(dtype=torch.float32)

        if tracks.shape[0] != images.shape[0]:
            raise ValueError(
                "Number of frames in images does not match tracked trajectories "
                f"({images.shape[0]=} vs {tracks.shape[0]=})."
            )

        visibility_mask = (visibilities > visibility_threshold).to(dtype=torch.bool)

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        from vggt.utils.visual_track import visualize_tracks_on_images  # deferred import

        visualize_tracks_on_images(
            images=images,
            tracks=tracks,
            track_vis_mask=visibility_mask,
            out_dir=str(output_dir_path),
            image_format="CHW",
            normalize_mode="[0,1]",
            frames_per_row=frames_per_row,
            save_grid=save_grid,
        )


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


def _high_confidence_pointcloud(config: VggtConfiguration, vggt_output: VggtOutput) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw VGGT predictions into point attributes."""
    points_3d = vggt_output.dense_points.to(torch.float32).cpu().numpy()
    points_rgb = (vggt_output.images.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

    depth_conf_np = vggt_output.depth_confidence.to(torch.float32).cpu().numpy()
    conf_threshold = min(config.confidence_threshold, depth_conf_np.mean() - depth_conf_np.std())
    conf_mask = depth_conf_np >= conf_threshold
    conf_mask = randomly_limit_trues(conf_mask, config.max_num_points)  # limit number of points if asked
    return points_3d[conf_mask], points_rgb[conf_mask]


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


def _convert_vggt_outputs_to_gtsfm_data(
    *,
    vggt_output: VggtOutput,
    original_coords: torch.Tensor,
    image_indices: Sequence[int],
    image_names: Optional[Sequence[str]],
    config: VggtConfiguration,
    points_3d: np.ndarray,
    points_rgb: np.ndarray,
    tracking_result: VGGTTrackingResult | None = None,
) -> tuple[GtsfmData, GtsfmData | None]:
    """Convert raw VGGT predictions into ``GtsfmData``."""

    extrinsic_np = vggt_output.extrinsic.to(torch.float32).cpu().numpy()
    intrinsic_np = vggt_output.intrinsic.to(torch.float32).cpu().numpy()
    original_coords_np = original_coords.to(torch.float32).cpu().numpy()
    image_names_str = [str(name) for name in image_names] if image_names is not None else None

    gtsfm_data = GtsfmData(number_images=len(image_indices))

    for local_idx, global_idx in enumerate(image_indices):
        image_width = float(original_coords_np[local_idx, 4])
        image_height = float(original_coords_np[local_idx, 5])

        scaled_intrinsic = intrinsic_np[local_idx]

        camera = torch_utils.camera_from_matrices(extrinsic_np[local_idx], scaled_intrinsic)
        gtsfm_data.add_camera(global_idx, camera)  # type: ignore[arg-type]
        gtsfm_data.set_image_info(
            global_idx,
            name=image_names_str[local_idx] if image_names_str is not None else None,
            shape=(int(image_height), int(image_width)),
        )

    if tracking_result is None and points_3d.size > 0 and points_rgb is not None:
        for j, xyz in enumerate(points_3d):
            track = torch_utils.colored_track_from_point(xyz, points_rgb[j])
            gtsfm_data.add_track(track)

    if tracking_result:

        # track masks according to visibility, reprojection error, etc
        max_reproj_error = float(config.max_reproj_error)
        track_mask = tracking_result.visibilities > config.track_vis_thresh

        confidence_threshold = config.track_conf_thresh
        confidence_threshold = min(
            confidence_threshold, np.mean(tracking_result.confidences) + np.std(tracking_result.confidences)
        )
        if tracking_result.confidences is not None:
            track_mask = np.logical_and(track_mask, tracking_result.confidences > confidence_threshold)

        enforce_reproj_filter = (
            tracking_result.points_3d is not None and np.isfinite(max_reproj_error) and max_reproj_error > 0.0
        )

        inlier_num = track_mask.sum(0)
        min_measurements = 2
        valid_mask = inlier_num >= min_measurements  # a track is invalid if without two inliers
        valid_idx = np.nonzero(valid_mask)[0]

        logger.info("num points 3d: %d, num valid idx: %d", tracking_result.points_3d.shape[0], len(valid_idx))

        for valid_id in valid_idx:
            rgb: np.ndarray
            if tracking_result.colors is not None and valid_id < tracking_result.colors.shape[0]:
                rgb = tracking_result.colors[valid_id]
            elif points_rgb is not None and valid_id < points_rgb.shape[0]:
                rgb = points_rgb[valid_id]
            else:
                rgb = np.zeros(3, dtype=np.uint8)
            point_xyz = tracking_result.points_3d[valid_id]
            gtsam_point = Point3(float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]))
            per_track_measurements: list[tuple[int, float, float]] = []
            max_error_for_track = 0.0
            frame_idx = np.where(track_mask[:, valid_id])[0]
            for local_id in frame_idx:
                global_idx = image_indices[local_id]
                u, v = tracking_result.tracks[local_id, valid_id]
                camera = gtsfm_data.get_camera(global_idx)
                if not _is_point_in_front_of_camera(camera, point_xyz):
                    continue
                if enforce_reproj_filter:
                    projected = camera.project(gtsam_point)
                    proj_u = float(projected[0])
                    proj_v = float(projected[1])
                    reproj_err = float(np.hypot(u - proj_u, v - proj_v))
                    max_error_for_track = max(max_error_for_track, reproj_err)
                per_track_measurements.append((global_idx, u, v))

            if len(per_track_measurements) < min_measurements:
                continue
            if enforce_reproj_filter and max_error_for_track > max_reproj_error:
                continue
            track = torch_utils.colored_track_from_point(point_xyz, rgb)
            for global_idx, float_u, float_v in per_track_measurements:
                track.addMeasurement(global_idx, Point2(float_u, float_v))
            min_triangulation_angle = config.min_triangulation_angle
            if min_triangulation_angle > 0.0:
                import gtsfm.utils.tracks as track_utils  # local import to avoid heavier dependency at module load

                cameras: dict[int, gtsfm_types.CAMERA_TYPE] = {}
                for global_idx, _, _ in per_track_measurements:
                    camera = gtsfm_data.get_camera(global_idx)
                    cameras[global_idx] = camera
                if track_utils.get_max_triangulation_angle(track, cameras) < min_triangulation_angle:
                    continue
            gtsfm_data.add_track(track)

    gtsfm_data_pre_ba: GtsfmData | None = None
    if config.run_bundle_adjustment_on_leaf:
        if config.store_pre_ba_result:
            gtsfm_data_pre_ba = gtsfm_data
        if gtsfm_data.number_tracks() == 0:
            logger.warning("Skipping bundle adjustment because VGGT produced no valid tracks.")
        else:
            try:
                gtsfm_data, should_run_ba = data_utils.remove_cameras_with_no_tracks(gtsfm_data, "node-level BA")
                if not should_run_ba:
                    return gtsfm_data, gtsfm_data_pre_ba
                optimizer = BundleAdjustmentOptimizer()
                gtsfm_data_with_ba, _ = optimizer.run_simple_ba(gtsfm_data, verbose=False)
                return gtsfm_data_with_ba, gtsfm_data_pre_ba
            except Exception as exc:
                logger.warning("⚠️ Failed to run bundle adjustment: %s", exc)

    return gtsfm_data, gtsfm_data_pre_ba


def _offload_vggt_model(model: Optional[VGGT]) -> None:
    """Move the VGGT model back to CPU to free GPU memory for tracking."""
    if model is None or not torch.cuda.is_available():
        return
    try:
        model.to("cpu")
    except RuntimeError as exc:  # pragma: no cover - defensive
        logger.warning("Failed to offload VGGT model to CPU: %s", exc)
    torch.cuda.empty_cache()


def run_VGGT(
    images: torch.Tensor,
    *,
    config: Optional[VggtConfiguration] = None,
    model: Optional[VGGT] = None,
    weights_path: PathLike | None = None,
) -> VggtOutput:
    """Run VGGT and unproject depth using the geometry helper."""
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

    # FastVGGT requires the model to know the actual patch grid dimensions used for token merging.
    patch_w = max(1, images.shape[-1] // getattr(model.aggregator, "patch_size", 14))
    patch_h = max(1, images.shape[-2] // getattr(model.aggregator, "patch_size", 14))
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
            batched = images.unsqueeze(0)  # make into (training) batch of 1
            tokens, ps_idx = model.aggregator(batched)  # transformer backbone
        with torch.cuda.amp.autocast(dtype=torch.float32):
            pose_enc = model.camera_head(tokens)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batched.shape[-2:])
            depth_map, depth_conf = model.depth_head(tokens, batched, ps_idx)

    depth_confidence = depth_conf.squeeze(0)
    if depth_confidence.ndim == 4 and depth_confidence.shape[-1] == 1:
        depth_confidence = depth_confidence.squeeze(-1)

    depth_map = depth_map.squeeze(0).to(dtype=torch.float32)
    extrinsic = extrinsic.squeeze(0).to(dtype=torch.float32)
    intrinsic = intrinsic.squeeze(0).to(dtype=torch.float32)
    dense_points_np = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    dense_points = torch.from_numpy(dense_points_np).to(device=resolved_device, dtype=torch.float32)

    return VggtOutput(
        device=resolved_device,
        dtype=resolved_dtype,
        images=images,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth_map=depth_map,
        depth_confidence=depth_confidence,
        dense_points=dense_points,
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


def _import_vggsfm_utils():
    """Return the vendored vggsfm utilities module from the VGGT submodule."""

    try:
        from vggt.dependency import vggsfm_utils as _vggsfm_utils  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when the submodule is absent
        if _USING_FASTVGGT:
            try:
                tracker_module = _import_from_vanilla_vggt("dependency.vggsfm_utils")
                logger.info("Using vggsfm utilities from the vanilla VGGT submodule.")
                return tracker_module  # type: ignore[return-value]
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


def _run_vggt_head_tracking(
    vggt_output: VggtOutput,
    *,
    model: VGGT,
    config: Optional[VggtConfiguration] = None,
) -> VGGTTrackingResult:
    """Generate dense feature tracks using the VGGT track head."""

    cfg = config or VggtConfiguration()
    vggsfm_utils = _import_vggsfm_utils()

    device = vggt_output.device
    if device.type != "cuda":
        raise RuntimeError(
            "VGGT tracking requires a CUDA-capable GPU because DINO relies on flash attention. "
            "Re-run the pipeline with CUDA available."
        )

    images = vggt_output.images
    if images.device != device or images.dtype != torch.float32:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)

    frame_num = images.shape[0]
    query_frame_indexes = vggsfm_utils.generate_rank_by_dino(
        images,
        query_frame_num=cfg.query_frame_num,
        image_size=518,
        model_name="dinov2_vitb14_reg",
        device=device,
        spatial_similarity=False,
    )
    if 0 in query_frame_indexes:
        query_frame_indexes.remove(0)
    query_frame_indexes = [0, *query_frame_indexes]

    extractors = vggsfm_utils.initialize_feature_extractors(
        max_query_num=cfg.max_query_pts,
        extractor_method=cfg.keypoint_extractor,
        device=device,
    )

    dense_points = vggt_output.dense_points
    depth_confidence = vggt_output.depth_confidence
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
        if query_points.shape[1] > cfg.max_query_pts:
            query_points = query_points[:, : cfg.max_query_pts]

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

        with torch.no_grad():
            with amp_autocast("cuda", dtype=vggt_output.dtype):
                aggregated_tokens_list, ps_idx = model.aggregator(reorder_images[None])
            if aggregated_tokens_list and aggregated_tokens_list[0].dtype != torch.float32:
                aggregated_tokens_list = [tokens.float() for tokens in aggregated_tokens_list]
            with amp_autocast("cuda", dtype=torch.float32):
                track_list, vis_scores, conf_scores = model.track_head(
                    aggregated_tokens_list,
                    reorder_images[None],
                    ps_idx,
                    query_points=query_points,
                )

        pred_track = track_list[-1]
        pred_track = pred_track.squeeze(0)
        vis_scores = vis_scores.squeeze(0)
        conf_scores = conf_scores.squeeze(0)
        reordered = vggsfm_utils.switch_tensor_order([pred_track, vis_scores, conf_scores], reorder_index, dim=0)
        pred_track, pred_vis, pred_conf_score = reordered

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
        if int(filtered_flag.sum().item()) > cfg.max_query_pts // 2:
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


def run_vggt_tracking(
    vggt_output: VggtOutput,
    *,
    config: Optional[VggtConfiguration] = None,
    model: Optional[VGGT] = None,
) -> VGGTTrackingResult:
    """Generate dense feature tracks using the configured VGGT tracking backend."""

    cfg = config or VggtConfiguration()
    if model is None:
        raise ValueError("VGGT tracking_head requires a loaded VGGT model.")
    return _run_vggt_head_tracking(vggt_output, model=model, config=cfg)


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
        images: Tensor shaped ``(num_frames, 3, H, W)`` at the VGGT load resolution.
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

    model_for_tracking = None
    if cfg.tracking and model_for_tracking is None:
        model_for_tracking = model

    vggt_output = run_VGGT(images, config=cfg, model=model, weights_path=weights_path)

    tracking_result = None
    if cfg.tracking:
        tracking_result = run_vggt_tracking(vggt_output, config=cfg, model=model_for_tracking)

    if cfg.tracking and vggt_output.device.type == "cuda":
        if model_for_tracking is not None:
            _offload_vggt_model(model_for_tracking)
        else:
            torch.cuda.empty_cache()

    points_3d, points_rgb = _high_confidence_pointcloud(cfg, vggt_output)

    gtsfm_data, gtsfm_data_pre_ba = _convert_vggt_outputs_to_gtsfm_data(
        config=cfg,
        vggt_output=vggt_output,
        original_coords=original_coords,
        image_indices=image_indices,
        image_names=image_names,
        points_3d=points_3d,
        points_rgb=points_rgb,
        tracking_result=tracking_result,
    )

    if vggt_output.device.type == "cuda":
        del vggt_output
        torch.cuda.empty_cache()

    return VggtReconstruction(
        gtsfm_data=gtsfm_data,
        pre_ba_data=gtsfm_data_pre_ba,
        points_3d=points_3d,
        points_rgb=points_rgb,
        tracking_result=tracking_result,
    )


def run_reconstruction_gtsfm_data_only(images: torch.Tensor, **kwargs) -> GtsfmData:
    """Run VGGT on a batch of images and convert outputs to ``GtsfmData``.

    Args:
        images: Tensor shaped ``(num_frames, 3, H, W)`` at the VGGT load resolution.
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
    "load_image_batch_vggt_loader",
    "load_and_preprocess_images_square",
    "resolve_weights_path",
    "load_model",
    "run_VGGT",
    "run_reconstruction",
    "run_vggt_tracking",
    "VGGTTrackingResult",
    "VggtOutput",
]
