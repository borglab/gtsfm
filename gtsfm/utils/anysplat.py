"""Helpers to integrate the AnySplat submodule with GTSFM."""

import sys
from dataclasses import dataclass
from pathlib import Path

import gtsam
import numpy as np
import torch
from gtsam import Point3, Pose3, Rot3

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


def pose_from_extrinsic(matrix: np.ndarray) -> Pose3:
    """Convert a Anysplat extrinsic matrix (camera-from-world) to a Pose3 (world-from-camera)."""
    cRw: np.ndarray = matrix[:3, :3]
    t = matrix[:3, 3]
    return Pose3(Rot3(cRw), Point3(*t)).inverse()


def calibration_from_intrinsic(matrix: np.ndarray, camera_type: str) -> gtsam.Cal3:
    """Map a 3x3 intrinsic matrix to the corresponding GTSAM calibration type."""
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    cx = float(matrix[0, 2])
    cy = float(matrix[1, 2])
    model = camera_type.upper()
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        return gtsam.Cal3Bundler(fx, 0.0, 0.0, cx, cy)
    return gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
