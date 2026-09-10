"""VGGT Omega implementation of the GeometryTransformer abstraction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms as TF

from gtsfm.frontend.geometry_transformer import GeometryTransformer, GeometryTransformerOutput
from gtsfm.utils import logger as logger_utils
from gtsfm.utils import torch as torch_utils

PathLike = Union[str, Path]

logger = logger_utils.get_logger()

# ---------------------------------------------------------------------------
# Submodule / path management
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRDPARTY_ROOT = REPO_ROOT / "thirdparty"
VGGT_OMEGA_SUBMODULE_PATH = THIRDPARTY_ROOT / "vggt-omega"
DEFAULT_WEIGHTS_PATH = VGGT_OMEGA_SUBMODULE_PATH / "weights" / "vggt_omega_1b_512.pt"


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


_ensure_submodule_on_path(VGGT_OMEGA_SUBMODULE_PATH, "vggt_omega")


def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    points3d: np.ndarray = np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )
    return points3d


from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

DEFAULT_FIXED_RESOLUTION_VGGT_OMEGA = 512


def load_image_batch_vggt_omega_loader(
    loader,
    indices: List[int],
    mode: str = "balanced",
    image_resolution: int = DEFAULT_FIXED_RESOLUTION_VGGT_OMEGA,
    patch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load loader-resized images and preprocess them for VGGT-Omega.

    ``original_coords`` maps pixels from ``loader.get_image()`` into the final
    padded Omega tensor. Each row is ``[left, top, right, bottom, scaled_w,
    scaled_h]`` and can be used as::

        u_omega = u_loader * scaled_w / loader_w - left
        v_omega = v_loader * scaled_h / loader_h - top

    Args:
        loader: Loader instance providing ``get_image``.
        indices: List of image indices to load.
        mode: ``balanced` keeps the total token count close to image_resolution**2.
            `max_size` resizes the longest side to image_resolution.
        image_resolution: Vggt-Omega preprocessing resolution.
        patch_size: Vggt-Omega image patch size.

    Returns:
        Batched images shaped ``(N, 3, H, W)`` and coordinates shaped ``(N, 6)``.
    """
    # checks from vggt-omega
    if mode not in ("balanced", "max_size"):
        raise ValueError("Mode must be either 'balanced' or 'max_size'")
    if image_resolution <= 0:
        raise ValueError("image_resolution must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if image_resolution % patch_size != 0:
        raise ValueError("image_resolution must be divisible by patch_size")

    images: list[torch.Tensor] = []
    transforms: list[tuple[int, int, int, int, int, int, int, int]] = []
    to_tensor = TF.ToTensor()

    for idx in indices:
        image = PILImage.fromarray(loader.get_image(idx).value_array).convert("RGB")
        loader_w, loader_h = image.size

        crop_x = 0
        crop_y = 0
        crop_w = loader_w
        crop_h = loader_h
        aspect_ratio = loader_h / max(loader_w, 1)
        if aspect_ratio < 0.5:
            crop_w = min(loader_w, max(1, int(round(loader_h / 0.5))))
            crop_x = max((loader_w - crop_w) // 2, 0)
        elif aspect_ratio > 2.0:
            crop_h = min(loader_h, max(1, int(round(loader_w * 2.0))))
            crop_y = max((loader_h - crop_h) // 2, 0)

        image = image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        cropped_aspect_ratio = crop_h / max(crop_w, 1)

        if mode == "balanced":
            num_patches = (image_resolution // patch_size) ** 2
            width_patches = max(1, int(np.round(np.sqrt(num_patches / cropped_aspect_ratio))))
            height_patches = max(1, int(np.round(num_patches / np.sqrt(num_patches / cropped_aspect_ratio))))
            target_w = width_patches * patch_size
            target_h = height_patches * patch_size
        elif cropped_aspect_ratio >= 1.0:
            target_h = image_resolution
            target_w = max(
                patch_size,
                int(np.round((image_resolution / cropped_aspect_ratio) / patch_size)) * patch_size,
            )
        else:
            target_w = image_resolution
            target_h = max(
                patch_size,
                int(np.round((image_resolution * cropped_aspect_ratio) / patch_size)) * patch_size,
            )

        image = image.resize((target_w, target_h), PILImage.Resampling.BICUBIC)
        images.append(to_tensor(image))
        transforms.append((loader_w, loader_h, crop_x, crop_y, crop_w, crop_h, target_w, target_h))

    batch_h = max(image.shape[1] for image in images)
    batch_w = max(image.shape[2] for image in images)
    padded_images: list[torch.Tensor] = []
    original_coords: list[list[float]] = []

    for image, transform in zip(images, transforms):
        loader_w, loader_h, crop_x, crop_y, crop_w, crop_h, target_w, target_h = transform
        pad_left = (batch_w - target_w) // 2
        pad_right = batch_w - target_w - pad_left
        pad_top = (batch_h - target_h) // 2
        pad_bottom = batch_h - target_h - pad_top
        padded_images.append(
            torch.nn.functional.pad(
                image,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        )

        resize_x = target_w / crop_w
        resize_y = target_h / crop_h
        scaled_w = loader_w * resize_x
        scaled_h = loader_h * resize_y

        # this is because the existing frontend expects u_omega = u_loader * resize_x - left
        # and natural mapping for this transform is u_omega = (u_loader - crop_x) * resize_x + pad_left
        left = crop_x * resize_x - pad_left
        top = crop_y * resize_y - pad_top
        original_coords.append([left, top, left + batch_w, top + batch_h, scaled_w, scaled_h])

    return torch.stack(padded_images), torch.tensor(original_coords, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def resolve_weights_path(weights_path: PathLike | None = None) -> Path:
    """Return a concrete path to the VGGT Omega checkpoint, validating that it exists."""
    checkpoint = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"VGGT Omega weights not found at {checkpoint}. Download weights via `scripts/download_model_weights.sh`."
        )
    return checkpoint


def load_model(
    device: torch.device,
):
    """Load the VGGT Omega model weights on the requested device."""
    if device.type != "cuda":
        raise RuntimeError("VGGT-Omega requires CUDA.")
    checkpoint = resolve_weights_path(DEFAULT_WEIGHTS_PATH)
    model = VGGTOmega()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    model.eval()
    model.to(device)
    return model


# Module-level singleton so each Dask worker loads the ~1B VGGT-Omega weights only ONCE. When this
# transformer is driven by ClusterVGGTWithFrontend with model_cache_key=False, the optimizer passes
# model=None for every cluster; without this cache that would reload the 1B model per cluster.
_OMEGA_MODEL_CACHE: dict[str, Any] = {}


def resolve_cached_model(device: torch.device):
    """Return a per-worker cached VGGT-Omega model, loading the weights once per device."""
    key = str(device)
    cached = _OMEGA_MODEL_CACHE.get(key)
    if cached is None:
        logger.info("⏳ Loading VGGT-Omega model weights (worker cache)...")
        cached = load_model(device)
        _OMEGA_MODEL_CACHE[key] = cached
        logger.info("✅ VGGT-Omega model weights loaded and cached.")
    return cached


# ---------------------------------------------------------------------------
# VggtOmegaGeometryTransformer
# ---------------------------------------------------------------------------


class VggtOmegaGeometryTransformer(GeometryTransformer):
    """Runs VGGT Omega model inference to predict poses, depths, and dense points."""

    def __init__(self):
        self.resolved_dtype = torch.float32

    def load_image_batch(self, loader, indices, mode: str = "balanced"):
        """Preprocess images with VGGT-Omega's own loader (512px, 16-patch aligned, aspect-cropped).

        Omega's modes are ``balanced``/``max_size`` (not VGGT's ``crop``/``pad``), so the optimizer's
        ``input_mode`` does not apply here — anything other than a valid omega mode falls back to
        ``balanced``. The ``original_coords`` returned match the build's keypoint->depth mapping, so the
        depth lookup stays consistent with omega's depth-map output resolution.
        """
        if mode not in ("balanced", "max_size"):
            mode = "balanced"
        return load_image_batch_vggt_omega_loader(loader, indices, mode=mode)

    def predict(
        self,
        images: torch.Tensor,
        *,
        model: Optional[VGGTOmega] = None,
        **kwargs: Any,
    ) -> GeometryTransformerOutput:
        """Run VGGT Omega forward pass and return unified output.

        Args:
            images: Tensor shaped ``(N, 3, H, W)``.

        Returns:
            class:`GeometryTransformerOutput` with poses, depths, and dense points.
        """
        self.resolved_device = torch_utils.default_device()
        images = images.to(self.resolved_device, dtype=self.resolved_dtype)
        if model is None:
            model = resolve_cached_model(self.resolved_device)
        else:
            model = model.to(self.resolved_device)
            assert model is not None
            model.eval()

        with torch.inference_mode():
            predictions = model(images)

        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

        depth = predictions["depth"]
        depth_conf = predictions["depth_conf"]

        dense_points = unproject_depth_map_to_point_map(
            depth.squeeze(0).detach().float().cpu().numpy(),
            extrinsics.squeeze(0).detach().float().cpu().numpy(),
            intrinsics.squeeze(0).detach().float().cpu().numpy(),
        )

        return GeometryTransformerOutput(
            device=self.resolved_device,
            dtype=self.resolved_dtype,
            images=images,
            extrinsic=extrinsics.squeeze(0),
            intrinsic=intrinsics.squeeze(0),
            depth_map=depth.squeeze(0).squeeze(-1),
            depth_confidence=depth_conf.squeeze(0),
            dense_points=torch.from_numpy(dense_points).to(
                self.resolved_device,
                dtype=self.resolved_dtype,
            ),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_WEIGHTS_PATH",
    "REPO_ROOT",
    "THIRDPARTY_ROOT",
    "VGGT_OMEGA_SUBMODULE_PATH",
    "VggtOmegaGeometryTransformer",
    "_ensure_submodule_on_path",
    "load_image_batch_vggt_omega_loader",
    "load_model",
    "resolve_weights_path",
]
