"""Helpers to integrate the AnySplat submodule with GTSFM."""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils import logger as logger_utils

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

try:
    from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA  # type: ignore
    from src.model.types import Gaussians  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The 'anysplat' Python package could not be imported even after adding the submodule to sys.path."
    ) from exc


@dataclass
class AnySplatReconstructionResult:
    """Outputs from the Anysplat generate splats function."""

    gtsfm_data: GtsfmData
    splats: Gaussians
    pred_context_pose: dict[str, torch.Tensor]
    height: int
    width: int
    decoder: DecoderSplattingCUDA
