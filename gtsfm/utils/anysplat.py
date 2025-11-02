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

try:  # pragma: no cover - exercised by integration tests, hard to simulate in unit tests.
    from src.misc.image_io import save_interpolated_video as _save_interpolated_video_impl  # type: ignore
    from src.model.decoder.decoder_splatting_cuda import (
        DecoderSplattingCUDA as _DecoderSplattingCUDAImpl,
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


def _require_thirdparty() -> None:
    """Ensure the AnySplat thirdparty dependencies imported successfully."""

    if _IMPORT_ERROR is not None:
        raise ImportError(
            "The 'anysplat' Python package could not be imported even after adding the submodule to sys.path."
        ) from _IMPORT_ERROR


def save_interpolated_video(*args: Any, **kwargs: Any) -> Any:
    """Proxy to AnySplat's video export helper, ensuring dependencies are present."""

    _require_thirdparty()
    assert _SAVE_INTERPOLATED_VIDEO is not None
    return _SAVE_INTERPOLATED_VIDEO(*args, **kwargs)


def export_ply(*args: Any, **kwargs: Any) -> Any:
    """Proxy to AnySplat's PLY export helper, ensuring dependencies are present."""

    _require_thirdparty()
    assert _EXPORT_PLY is not None
    return _EXPORT_PLY(*args, **kwargs)


@dataclass
class AnySplatReconstructionResult:
    """Outputs from the Anysplat generate splats function."""

    gtsfm_data: GtsfmData
    splats: GaussiansProtocol
    pred_context_pose: dict[str, torch.Tensor]
    height: int
    width: int
    decoder: DecoderSplattingCUDAProtocol


def load_model(
    *,
    device: torch.device | str | None = None,
    local_files_only: bool = False,
    checkpoint_path: str | Path | None = None,
) -> Any:
    """Load AnySplat weights optionally moving the model to the requested device."""

    _require_thirdparty()
    assert AnySplat is not None

    target = checkpoint_path or "lhjiang/anysplat"
    load_kwargs: dict[str, Any] = {}
    if checkpoint_path is None:
        load_kwargs["local_files_only"] = local_files_only
    else:
        load_kwargs["local_files_only"] = True

    model = AnySplat.from_pretrained(target, **load_kwargs)
    model.eval()

    if device is not None:
        resolved_device = torch.device(device)
        model = model.to(resolved_device)
    return model
