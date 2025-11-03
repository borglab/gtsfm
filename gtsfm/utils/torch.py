"""Common pytorch utilities."""

from typing import Optional, Sequence, Union

import gtsam  # type: ignore
import numpy as np
import torch
from gtsam import Point3, Pose3, Rot3, SfmTrack

import gtsfm.common.types as gtsfm_types


def default_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve a concrete device for PyTorch inference."""
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
    """Convert an extrinsic matrix (camera-from-world) to a Pose3 (world-from-camera)."""
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


def camera_from_matrices(extrinsic: np.ndarray, intrinsic: np.ndarray) -> gtsfm_types.CAMERA_TYPE:
    """Instantiate a Pinhole camera from raw extrinsic/intrinsic matrices."""
    calibration = calibration_from_intrinsic(intrinsic)
    pose = pose_from_extrinsic(extrinsic)
    return gtsfm_types.create_camera(pose, calibration)


def colored_track_from_point(
    point_xyz: np.ndarray,
    rgb: Optional[Sequence[float]] = None,
) -> SfmTrack:
    """Construct an SfmTrack with optional RGB color attributes."""
    coords = np.asarray(point_xyz, dtype=np.float64)
    track = SfmTrack(Point3(*coords))
    if rgb is not None:
        color = np.asarray(rgb, dtype=np.float64)
        track.r = float(color[0])
        track.g = float(color[1])
        track.b = float(color[2])
    return track
