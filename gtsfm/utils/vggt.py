"""Helpers to integrate the VGGT submodule with GTSFM.

This module centralizes everything needed to import Facebook's VGGT code that
ships as a Git submodule in ``thirdparty/vggt``.  It exposes small utilities to
resolve the checkpoint path, pick sane default devices/dtypes, and lazily load
the model so that other parts of GTSFM do not have to duplicate this boilerplate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import torch

from gtsfm.utils import logger as logger_utils

PathLike = Union[str, Path]

_LOGGER = logger_utils.get_logger()

REPO_ROOT = Path(__file__).resolve().parents[2]
VGGT_SUBMODULE_PATH = REPO_ROOT / "thirdparty" / "vggt"
DEFAULT_WEIGHTS_PATH = VGGT_SUBMODULE_PATH / "weights" / "model.pt"


def _ensure_vggt_repo_on_path() -> None:
    """Add the VGGT submodule to ``sys.path`` so we can import it like a package."""
    if not VGGT_SUBMODULE_PATH.exists():
        raise ImportError(
            f"VGGT submodule not found at {VGGT_SUBMODULE_PATH}. "
            "Did you run 'git submodule update --init --recursive'?"
        )

    repo_str = str(VGGT_SUBMODULE_PATH)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


_ensure_vggt_repo_on_path()

try:
    from vggt.models.vggt import VGGT  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The 'vggt' Python package could not be imported even after adding the submodule to sys.path."
    ) from exc


def resolve_vggt_weights_path(checkpoint_path: PathLike | None = None) -> Path:
    """Return a concrete path to the VGGT checkpoint, validating that it exists."""
    path = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_WEIGHTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"VGGT checkpoint not found at {path}. "
            "Please run 'bash download_model_weights.sh' from the repo root."
        )
    return path


def default_vggt_device(device: torch.device | str | None = None) -> torch.device:
    """Pick a reasonable default device for VGGT inference."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def default_vggt_dtype(device: torch.device | None = None) -> torch.dtype:
    """Select a dtype that matches the provided device (VGGT prefers fp16/bfloat16 on Ada+ GPUs)."""
    if device is None:
        device = default_vggt_device()

    if device.type == "cuda" and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def load_vggt_model(
    checkpoint_path: PathLike | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    strict: bool = True,
) -> VGGT:
    """Instantiate VGGT, load weights, move it to the requested device/dtype, and return it."""
    resolved_device = default_vggt_device(device)
    resolved_dtype = dtype
    if resolved_dtype is None and resolved_device.type == "cuda":
        resolved_dtype = default_vggt_dtype(resolved_device)

    weights_path = resolve_vggt_weights_path(checkpoint_path)
    _LOGGER.info("⏳ Loading VGGT checkpoint from %s", weights_path)

    model = VGGT()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

    model = model.to(resolved_device)
    if resolved_dtype is not None:
        # dtype casting is only attempted when explicitly requested or inferred for CUDA devices.
        model = model.to(dtype=resolved_dtype)

    return model


__all__ = [
    "VGGT_SUBMODULE_PATH",
    "DEFAULT_WEIGHTS_PATH",
    "resolve_vggt_weights_path",
    "default_vggt_device",
    "default_vggt_dtype",
    "load_vggt_model",
]
