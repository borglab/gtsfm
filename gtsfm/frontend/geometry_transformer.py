"""Base abstraction for geometry transformer models (VGGT, AnySplat, etc.)."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from gtsfm.utils import logger as logger_utils

logger = logger_utils.get_logger()


@dataclass(frozen=True)
class ImagePlacement:
    """Where one preprocessed image sits in its model frame, before batch padding.

    ``left``/``top`` place the LOADER image's origin in the model frame, and ``scaled_w``/``scaled_h``
    are the loader image's dimensions after the model's resize, so a loader pixel maps into the model
    frame as::

        u_model = u_loader * scaled_w / loader_w - left
        v_model = v_loader * scaled_h / loader_h - top

    Each model-specific loader (VGGT, VGGT-Omega) computes its own resize/crop policy and reports the
    result as one of these; batch padding and the packed ``original_coords`` rows are then derived in
    exactly one place, :func:`assemble_image_batch`.
    """

    left: float
    top: float
    scaled_w: float
    scaled_h: float


def assemble_image_batch(
    images: list[torch.Tensor], placements: list[ImagePlacement]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad per-image tensors to a common shape, stack them, and derive ``original_coords``.

    The single source of the ``original_coords`` contract: after centering image ``i`` in the common
    batch frame (fill value 1.0), its row is::

        [left, top, left + batch_w, top + batch_h, scaled_w, scaled_h]

    with ``left``/``top`` shifted by the applied padding so the :class:`ImagePlacement` mapping formula
    stays valid in the batch frame.
    """
    batch_h = max(image.shape[1] for image in images)
    batch_w = max(image.shape[2] for image in images)
    shapes = {(image.shape[1], image.shape[2]) for image in images}
    if len(shapes) > 1:
        logger.warning("Found images with different shapes: %s", shapes)

    padded_images: list[torch.Tensor] = []
    rows: list[list[float]] = []
    for image, placement in zip(images, placements):
        pad_left = (batch_w - image.shape[2]) // 2
        pad_right = batch_w - image.shape[2] - pad_left
        pad_top = (batch_h - image.shape[1]) // 2
        pad_bottom = batch_h - image.shape[1] - pad_top
        if pad_left or pad_right or pad_top or pad_bottom:
            image = torch.nn.functional.pad(
                image, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
        left = placement.left - pad_left
        top = placement.top - pad_top
        rows.append([left, top, left + batch_w, top + batch_h, placement.scaled_w, placement.scaled_h])
        padded_images.append(image)

    return torch.stack(padded_images), torch.tensor(rows, dtype=torch.float32)


@dataclass
class GeometryTransformerConfig:
    """Shared configuration for geometry transformer models."""

    seed: int = 42
    confidence_threshold: float = 5.0
    max_num_points: int = 100000
    dtype: Optional[Union[str, torch.dtype]] = None


@dataclass
class GeometryTransformerOutput:
    """Unified output from a geometry transformer forward pass.

    Core fields are populated by all implementations (VGGT, AnySplat, etc.).
    Optional fields accommodate model-specific outputs like Gaussian splats.
    """

    device: torch.device
    dtype: torch.dtype
    images: torch.Tensor  # (N, 3, H, W)
    extrinsic: torch.Tensor  # (N, 3, 4) or (N, 4, 4)
    intrinsic: torch.Tensor  # (N, 3, 3)
    depth_map: torch.Tensor  # (N, H, W)
    depth_confidence: torch.Tensor  # (N, H, W)
    dense_points: torch.Tensor  # (N, H, W, 3)

    # Optional Gaussian splatting outputs (populated by AnySplat).
    splats: Optional[Any] = None
    decoder: Optional[Any] = None
    gaussian_metadata: Optional[dict[str, Any]] = None


class GeometryTransformer(abc.ABC):
    """Abstract base class for models that predict geometry from multi-view images.

    Subclasses (e.g. VggtGeometryTransformer) implement :meth:`predict` to run
    model inference and return a :class:`GeometryTransformerOutput`.
    """

    @abc.abstractmethod
    def predict(
        self,
        images: torch.Tensor,
        **kwargs: Any,
    ) -> GeometryTransformerOutput:
        """Run model inference on a batch of images.

        Args:
            images: Tensor shaped ``(N, 3, H, W)``.
            **kwargs: Implementation-specific options (e.g. ``model``, ``weights_path``).

        Returns:
            A :class:`GeometryTransformerOutput` containing poses, depths, points, etc.
        """

    def load_image_batch(self, loader, indices, mode: str = "crop"):
        """Load + preprocess a batch of images for this geometry model.

        Image preprocessing (resolution, patch alignment, cropping) is model-specific, and the
        per-pixel depth lookup downstream indexes the model's depth map using the ``original_coords``
        returned here — so the loader MUST match the resolution the model outputs. The default is the
        VGGT loader; models with different preprocessing (e.g. VGGT-Omega) override this.

        Args:
            loader: GTSFM loader providing ``get_image``.
            indices: image indices to load.
            mode: preprocessing mode (VGGT: ``crop``/``pad``). Implementations may ignore it.

        Returns:
            ``(image_batch, original_coords)`` — the padded image tensor and (N, 6) crop/pad metadata.
        """
        from gtsfm.frontend.vggt_geometry_transformer import load_image_batch_vggt_loader

        return load_image_batch_vggt_loader(loader, indices, mode=mode)
