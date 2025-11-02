"""Common pytorch utilities."""

from typing import Optional, Union

import gtsam  # type: ignore
import numpy as np
import torch
from gtsam import Point3, Pose3, Rot3


def default_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve a concrete device for VGGT inference."""
    if device is not None:
        return torch.device(device)

    # Prefer CUDA if available, then MPS (Mac GPU), then fall back to CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def pose_from_extrinsic(matrix: np.ndarray) -> Pose3:
    """Convert a VGGT extrinsic matrix (camera-from-world) to a Pose3 (world-from-camera)."""
    cRw: np.ndarray = matrix[:3, :3]
    t = matrix[:3, 3]
    return Pose3(Rot3(cRw), Point3(*t)).inverse()


def calibration_from_intrinsic(matrix: np.ndarray) -> gtsam.Cal3_S2:
    """Map a 3x3 intrinsic matrix to a Cal3_S2."""
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    cx = float(matrix[0, 2])
    cy = float(matrix[1, 2])
    return gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
