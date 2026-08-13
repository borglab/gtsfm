"""VGGT implementation of the GeometryTransformer abstraction."""

from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image as PILImage
from torch.amp import autocast as amp_autocast  # type: ignore
from torchvision import transforms as TF

from gtsfm.frontend.geometry_transformer import (
    GeometryTransformer,
    GeometryTransformerConfig,
    GeometryTransformerOutput,
)
from gtsfm.utils import logger as logger_utils
from gtsfm.utils import torch as torch_utils

PathLike = Union[str, Path]

logger = logger_utils.get_logger()

# ---------------------------------------------------------------------------
# Submodule / path management
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

DEFAULT_FIXED_RESOLUTION = 518


def load_image_batch_vggt_loader(loader, indices: List[int], mode="crop"):
    """Load and preprocess images for VGGT model input.

    Args:
        loader: Loader instance providing ``get_image``.
        indices: List of image indices to load.
        mode: Preprocessing mode, either "crop" or "pad".

    Returns:
        Tuple of (images tensor shaped (N, 3, H, W), original_coords tensor shaped (N, 6)).
    """
    if len(indices) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518
    coords = []

    for idx in indices:
        img = loader.get_image(idx).value_array
        img = PILImage.fromarray(img)
        width, height = img.size

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:  # mode == "crop"
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), PILImage.Resampling.BICUBIC)
        img = to_tensor(img)

        coord = np.array([0.0, 0.0, float(new_width), float(new_height), float(new_width), float(new_height)])

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]
            coord[1] = start_y
            coord[3] = start_y + target_size
        elif mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                pad_left = max(0, pad_left)
                pad_right = max(0, pad_right)
                pad_top = max(0, pad_top)
                pad_bottom = max(0, pad_bottom)
                coord[0] = -pad_left
                coord[1] = -pad_top
                coord[2] = pad_right + img.shape[2]
                coord[3] = pad_bottom + img.shape[1]
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        coords.append(coord)

    if len(shapes) > 1:
        logger.warning("Found images with different shapes: %s", shapes)
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        padded_images = []
        padded_coords = []
        for img, coord in zip(images, coords):
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
                coord[0] = coord[0] - pad_left
                coord[1] = coord[1] - pad_top
                coord[2] = coord[2] + pad_right
                coord[3] = coord[3] + pad_bottom
            padded_coords.append(coord)
            padded_images.append(img)
        images = padded_images
        coords = padded_coords

    images = torch.stack(images)
    coords = np.array(coords)
    if len(indices) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)

    original_coords_tensor = torch.from_numpy(coords).float()
    return images, original_coords_tensor


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def default_dtype(device: torch.device) -> torch.dtype:
    """Pick a floating-point dtype suitable for VGGT on the provided device."""
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device=device)
        return torch.bfloat16 if capability[0] >= 8 else torch.float16
    elif device.type == "mps":
        return torch.float32
    else:
        return torch.float32


def resolve_weights_path(weights_path: PathLike | None = None) -> Path:
    """Return a VGGT checkpoint, downloading the default on first use."""
    checkpoint = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
    if checkpoint.exists():
        return checkpoint
    if weights_path is None:
        from huggingface_hub import hf_hub_download

        logger.info("⬇️ Downloading VGGT weights on first use...")
        return Path(hf_hub_download(repo_id="facebook/VGGT-1B", filename="model.pt"))
    raise FileNotFoundError(f"VGGT checkpoint not found at {checkpoint}.")


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
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


def offload_vggt_model(model: Optional[VGGT]) -> None:
    """Move the VGGT model back to CPU to free GPU memory."""
    if model is None or not torch.cuda.is_available():
        return
    try:
        model.to("cpu")
    except RuntimeError as exc:
        logger.warning("Failed to offload VGGT model to CPU: %s", exc)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Pointcloud extraction
# ---------------------------------------------------------------------------


def high_confidence_pointcloud(
    geo_output: GeometryTransformerOutput,
    confidence_threshold: float = 5.0,
    max_num_points: int = 100000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a filtered point cloud from geometry transformer output.

    Returns:
        Tuple of (points_3d, points_rgb) arrays.
    """
    points_3d = geo_output.dense_points.to(torch.float32).cpu().numpy()
    points_rgb = (geo_output.images.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

    depth_conf_np = geo_output.depth_confidence.to(torch.float32).cpu().numpy()
    conf_threshold = min(confidence_threshold, depth_conf_np.mean() - depth_conf_np.std())
    conf_mask = depth_conf_np >= conf_threshold
    conf_mask = randomly_limit_trues(conf_mask, max_num_points)
    return points_3d[conf_mask], points_rgb[conf_mask]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VggtGeometryConfig(GeometryTransformerConfig):
    """VGGT-specific geometry configuration."""

    model_ctor_kwargs: dict[str, Any] = field(default_factory=dict)
    use_sparse_attention: bool = False


# ---------------------------------------------------------------------------
# VggtGeometryTransformer
# ---------------------------------------------------------------------------


class VggtGeometryTransformer(GeometryTransformer):
    """Runs VGGT model inference to predict poses, depths, and dense points."""

    def __init__(self, config: VggtGeometryConfig | None = None) -> None:
        self.config = config or VggtGeometryConfig()

    def predict(
        self,
        images: torch.Tensor,
        *,
        model: Optional[VGGT] = None,
        weights_path: PathLike | None = None,
        config: VggtGeometryConfig | None = None,
    ) -> GeometryTransformerOutput:
        """Run VGGT forward pass and return unified output.

        Args:
            images: Tensor shaped ``(N, 3, H, W)``.
            model: Optional pre-loaded VGGT model.
            weights_path: Optional path to VGGT checkpoint.
            config: Optional override config.

        Returns:
            :class:`GeometryTransformerOutput` with poses, depths, and dense points.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("VGGT expects images shaped as (N, 3, H, W).")

        cfg = config or self.config
        resolved_device = torch_utils.default_device()
        requested_dtype = _resolve_dtype_argument(cfg.dtype)
        resolved_dtype = requested_dtype or default_dtype(resolved_device)
        logger.info(
            "VGGT inference dtype: cfg.dtype=%s requested=%s resolved=%s device=%s",
            str(cfg.dtype),
            str(requested_dtype),
            str(resolved_dtype),
            str(resolved_device),
        )

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
            assert model is not None
            if resolved_dtype is not None:
                model = model.to(dtype=resolved_dtype)
            assert model is not None
            model.eval()

        assert model is not None
        images = images.to(resolved_device, dtype=resolved_dtype)

        patch_w = max(1, images.shape[-1] // getattr(model.aggregator, "patch_size", 14))
        patch_h = max(1, images.shape[-2] // getattr(model.aggregator, "patch_size", 14))
        if hasattr(model, "update_patch_dimensions"):
            try:
                model.update_patch_dimensions(patch_w, patch_h)
            except Exception as exc:
                logger.warning("Failed to update VGGT patch dimensions (%dx%d): %s", patch_w, patch_h, exc)

        if resolved_device.type == "cuda":
            autocast_ctx: Any = amp_autocast("cuda", dtype=resolved_dtype)
        else:
            autocast_ctx = nullcontext()

        with torch.no_grad():
            with autocast_ctx:
                batched = images.unsqueeze(0)
                tokens, ps_idx = model.aggregator(batched)
            if resolved_device.type == "cuda":
                pose_depth_autocast_ctx: Any = torch.amp.autocast("cuda", dtype=torch.float32)
            else:
                pose_depth_autocast_ctx = nullcontext()
            with pose_depth_autocast_ctx:
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

        return GeometryTransformerOutput(
            device=resolved_device,
            dtype=resolved_dtype,
            images=images,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            depth_map=depth_map,
            depth_confidence=depth_confidence,
            dense_points=dense_points,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_FIXED_RESOLUTION",
    "DEFAULT_WEIGHTS_PATH",
    "FASTVGGT_SUBMODULE_PATH",
    "LIGHTGLUE_SUBMODULE_PATH",
    "REPO_ROOT",
    "THIRDPARTY_ROOT",
    "VGGT_SUBMODULE_PATH",
    "VggtGeometryConfig",
    "VggtGeometryTransformer",
    "_USING_FASTVGGT",
    "_ensure_submodule_on_path",
    "_import_from_vanilla_vggt",
    "_resolve_dtype_argument",
    "default_dtype",
    "high_confidence_pointcloud",
    "load_image_batch_vggt_loader",
    "load_and_preprocess_images_square",
    "load_model",
    "offload_vggt_model",
    "resolve_weights_path",
]
