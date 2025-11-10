"""Helpers to integrate the AnySplat submodule with GTSFM."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import torch

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils import logger as logger_utils
from gtsfm.utils.splat import GaussiansProtocol


class DecoderSplattingCUDAProtocol(Protocol):
    """Type protocol for AnySplat DecoderSplattingCUDA to satisfy mypy."""

    def cpu(self) -> DecoderSplattingCUDAProtocol:
        """Move the decoder to CPU."""
        ...

    def to(self, device: torch.device | str) -> DecoderSplattingCUDAProtocol:
        """Move the decoder to the specified device."""
        ...


logger = logger_utils.get_logger()


REPO_ROOT = Path(__file__).resolve().parents[2]
THIRDPARTY_ROOT = REPO_ROOT / "thirdparty"
ANYSPLAT_SUBMODULE_PATH = THIRDPARTY_ROOT / "AnySplat"


def _ensure_submodule_on_path(path: Path, name: str) -> None:
    """Add a vendored thirdparty module to ``sys.path`` if needed."""
    if not path.exists():
        raise ImportError(
            f"Required submodule '{name}' not found at {path}. " "Run 'git submodule update --init --recursive'?"
        )

    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_ensure_submodule_on_path(ANYSPLAT_SUBMODULE_PATH, "anysplat")

_SAVE_INTERPOLATED_VIDEO: Callable[..., Any] | None = None
_EXPORT_PLY: Callable[..., Any] | None = None
AnySplat: type[Any] | None = None  # type: ignore[assignment]
DecoderSplattingCUDA: type[DecoderSplattingCUDAProtocol] | Any = Any  # type: ignore[assignment]
Gaussians: type[GaussiansProtocol] | Any = Any  # type: ignore[assignment]
_IMPORT_ERROR: Exception | None = None
batchify_unproject_depth_map_to_point_map: Callable[..., torch.Tensor] | None = None

try:  # pragma: no cover - exercised by integration tests, hard to simulate in unit tests.
    from src.misc.image_io import save_interpolated_video as _save_interpolated_video_impl  # type: ignore
    from src.model.decoder.decoder_splatting_cuda import (
        DecoderSplattingCUDA as _DecoderSplattingCUDAImpl,
    )  # type: ignore
    from src.model.encoder.vggt.utils.geometry import (
        batchify_unproject_depth_map_to_point_map as _batchify_unproject_impl,
    )  # type: ignore
    from src.model.model.anysplat import AnySplat as _AnySplatImpl  # type: ignore
    from src.model.ply_export import export_ply as _export_ply_impl  # type: ignore
    from src.model.types import Gaussians as _GaussiansImpl  # type: ignore
except (ModuleNotFoundError, OSError) as exc:  # pragma: no cover - import guard
    _IMPORT_ERROR = exc
else:
    _SAVE_INTERPOLATED_VIDEO = _save_interpolated_video_impl
    _EXPORT_PLY = _export_ply_impl
    AnySplat = _AnySplatImpl  # type: ignore[assignment]
    DecoderSplattingCUDA = _DecoderSplattingCUDAImpl  # type: ignore[assignment]
    Gaussians = _GaussiansImpl  # type: ignore[assignment]
    batchify_unproject_depth_map_to_point_map = _batchify_unproject_impl


print(batchify_unproject_depth_map_to_point_map)