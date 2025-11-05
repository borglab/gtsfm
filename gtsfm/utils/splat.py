"""Utility functions for 3D Gaussian Splatting.

Authors: Harneet Singh Khanuja
"""

import math
import random
from dataclasses import is_dataclass, replace
from typing import Literal, Protocol, Tuple

import cv2
import gtsam  # type: ignore
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors  # type: ignore

from gtsfm.utils import logger as logger_utils

logger = logger_utils.get_logger()


class GaussiansProtocol(Protocol):
    """Type protocol for Gaussian splats to satisfy mypy."""

    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor
    covariances: torch.Tensor


# See https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/camera_utils.py
def get_rotation_matrix_from_two_vectors(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """
    Get the rotation matrix that rotates vec1 to vec2.

    Args:
        vec1: source vector
        vec2: target vector

    Returns
        The rotation matrix that rotates vec1 to vec2.
    """
    a = vec1 / torch.linalg.norm(vec1)
    b = vec2 / torch.linalg.norm(vec2)
    v = torch.linalg.cross(a, b)

    eps = 1e-6
    if torch.sum(torch.abs(v)) < eps:
        x = torch.tensor([1.0, 0, 0]) if abs(a[0]) < eps else torch.tensor([0, 1.0, 0])
        v = torch.linalg.cross(a, x)

    v = v / torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    theta = torch.acos(torch.clip(torch.dot(a, b), -1, 1))

    # Rodrigues rotation formula. https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    return torch.eye(3) + torch.sin(theta) * skew_sym_mat + (1 - torch.cos(theta)) * (skew_sym_mat @ skew_sym_mat)


# See https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/camera_utils.py
def auto_orient_and_center_poses(
    poses: torch.Tensor,
    method: str | Literal["up", "none"] = "none",
    center_method: str | Literal["poses", "none"] = "poses",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orients and centers the poses.
    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_method: The method to use to center the poses.

    Returns:
        poses: oriented and centered poses
        transform: the transformation matrix to orient and center (will be used for the SfM points)
    """
    origins = poses[..., :3, 3]
    mean_origin = torch.mean(origins, dim=0)

    translation = torch.zeros_like(mean_origin)
    if center_method == "poses":
        translation = mean_origin
    elif center_method == "none":
        pass
    else:
        raise ValueError(f"Unknown center method: {center_method}")

    R = torch.eye(3, device=poses.device)
    if method == "up":
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        R = get_rotation_matrix_from_two_vectors(up, torch.tensor([0.0, 1.0, 0.0], device=up.device))
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown orientation method: {method}")

    transform = torch.cat([R, R @ -translation[..., None]], dim=-1)
    poses_new = transform.to(poses.device) @ poses

    return poses_new, transform


# See https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/utils/dataloaders.py
def _undistort_image(distortion_params: np.ndarray, image: np.ndarray, K: np.ndarray):
    """
    Undistorts an image

    Args:
        distortion_params: the undistortion parameters
        image: image to be undistorted
        K: camera intrinsics matrix

    Returns:
        K: updated camera intrinsics matrix
        image: undistorted image
    """
    assert (
        distortion_params[3] == 0
    ), "We don't support the 4th Brown parameter for image undistortion, Only k1, k2, k3, p1, p2 can be non-zero."
    # we rearrange the distortion parameters because OpenCV expects the order (k1, k2, p1, p2, k3)
    # see https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    # The last two zeros are for k4 and k5, which are not used in this context.
    distortion_params_cv = np.array(
        [
            distortion_params[0],
            distortion_params[1],
            distortion_params[4],
            distortion_params[5],
            distortion_params[2],
            0,  # k4 is not used
            0,  # k5 is not used
            0,  # k6 is not used
        ]
    )

    # because OpenCV expects the pixel coord to be top-left, we need to shift the principal point by 0.5
    # see https://github.com/nerfstudio-project/nerfstudio/issues/3048
    K_shifted = K.copy()
    K_shifted[0, 2] -= 0.5
    K_shifted[1, 2] -= 0.5

    if np.any(distortion_params_cv):
        newK, roi = cv2.getOptimalNewCameraMatrix(K_shifted, distortion_params_cv, (image.shape[1], image.shape[0]), 1)
        image = cv2.undistort(image, K_shifted, distortion_params_cv, None, newK)
    else:
        newK = K_shifted
        roi = 0, 0, image.shape[1], image.shape[0]

    # crop the image and update the intrinsics accordingly
    x, y, w, h = roi
    image = image[y : y + h, x : x + w]
    newK[0, 2] -= x
    newK[1, 2] -= y

    newK[0, 2] += 0.5
    newK[1, 2] += 0.5
    K = newK

    return K, image


# See https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/cameras.py
def rescale_output_resolution(Ks, scaling_factor):
    """Rescale the output resolution of the cameras.

    Args:
        Ks: camera intrinsics matrix
        scaling_factor: Scaling factor.

    Returns:
        Rescaled camera intrinsics matrix
    """
    Ks[..., 0, 0] *= scaling_factor
    Ks[..., 1, 1] *= scaling_factor
    Ks[..., 0, 2] *= scaling_factor
    Ks[..., 1, 2] *= scaling_factor
    return Ks


def k_nearest_sklearn(x: torch.Tensor, k: int, metric: str = "euclidean"):
    """
    Find k-nearest neighbors using sklearn's NearestNeighbors.

    Args:
        x: input tensor
        k: number of neighbors to find
        metric: metric to use for distance computation

    Returns:
        distances: distances to the k-nearest neighbors
        indices: indices of the k-nearest neighbors
    """
    x_np = x.cpu().numpy()

    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric=metric, n_jobs=1).fit(x_np)

    distances, indices = nn_model.kneighbors(x_np)
    return torch.tensor(distances[:, 1:], dtype=torch.float32), torch.tensor(indices[:, 1:], dtype=torch.int64)


# See https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/utils/math.py
def random_quat_tensor(N: int):
    """
    Defines a random quaternion tensor.

    Args:
        N: Number of quaternions to generate

    Returns:
        a random quaternion tensor of shape (N, 4)

    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


# See https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/utils/spherical_harmonics.py
def num_sh_bases(degree: int) -> int:
    """
    Returns the number of spherical harmonic bases for a given degree.
    """
    MAX_SH_DEGREE = 4

    assert degree <= MAX_SH_DEGREE, f"We don't support degree greater than {MAX_SH_DEGREE}."
    return (degree + 1) ** 2


def set_random_seed(seed: int):
    """
    Setting random seed for making the program deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# See https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py
@torch.compile()
def get_viewmat(wTi_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of camera-to-world matrices to gsplat's world-to-camera format.
    This function is compiled with torch.compile for a speed boost.
    It converts to the gsplat standard ([Right, Down, Forward]).

    Args:
        wTi_tensor: A tensor of camera-to-world matrices with shape [N, 4, 4].
    Returns:
        A tensor of world-to-camera matrices with shape [N, 4, 4].
    """
    R = wTi_tensor[:, :3, :3]  # [N, 3, 3]
    T = wTi_tensor[:, :3, 3:4]  # [N, 3, 1]
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    viewmat[:, 3, 3] = 1.0
    return viewmat


def quaternion_to_matrix_xyzw(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions in (x, y, z, w) format to rotation matrices using GTSAM.

    Args:
        quaternions: tensor of shape (..., 4).

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    original_shape = quaternions.shape[:-1]
    flat_quats = quaternions.reshape(-1, 4).cpu().numpy()
    rotation_mats = []
    for q in flat_quats:
        wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=float)
        rot = gtsam.Rot3.Quaternion(*wxyz)
        rotation_mats.append(rot.matrix())
    rotation_mats = np.stack(rotation_mats, axis=0).reshape(*original_shape, 3, 3)
    return torch.tensor(rotation_mats, device=quaternions.device, dtype=quaternions.dtype)


def build_covariance_from_scales_quaternion(scales: torch.Tensor, rotations_xyzw: torch.Tensor) -> torch.Tensor:
    """
    Compute per-Gaussian covariance matrices via cov = R @ S @ S @ Rᵀ.

    Args:
        scales: tensor of shape (..., 3)
        rotations_xyzw: tensor of shape (..., 4)

    Returns:
        Covariance matrices of shape (..., 3, 3)
    """
    R = quaternion_to_matrix_xyzw(rotations_xyzw)
    S = torch.diag_embed(scales)
    cov = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
    return cov


def merge_gaussian_splats(
    gaussians_a: dict[str, torch.Tensor] | GaussiansProtocol,
    gaussians_b: dict[str, torch.Tensor] | GaussiansProtocol,
) -> dict[str, torch.Tensor] | GaussiansProtocol:
    """
    Concatenate two gaussian collections along the gaussian dimension.

    The inputs must both be dictionaries or both be dataclass implementations of :class:`GaussiansProtocol`.
    For dictionaries, all keys must match exactly. For protocol inputs, the batch dimension is assumed to be
    the leading axis (size ``B``) and the gaussian dimension at index 1. All tensors are concatenated along that
    gaussian axis and the same representation type is returned.
    """

    is_dict_input = isinstance(gaussians_a, dict)
    if is_dict_input != isinstance(gaussians_b, dict):
        raise TypeError("Both gaussian collections must share the same representation (dict or dataclass).")

    if is_dict_input:
        keys_a = set(gaussians_a.keys())
        keys_b = set(gaussians_b.keys())
        if keys_a != keys_b:
            raise ValueError("Gaussian dictionaries must contain identical keys to be merged.")

        merged: dict[str, torch.Tensor] = {}
        for key in gaussians_a:
            merged[key] = torch.cat([gaussians_a[key], gaussians_b[key]], dim=0)
        return merged

    if not is_dataclass(gaussians_a) or not is_dataclass(gaussians_b):
        raise TypeError("Expected dataclass instances implementing GaussiansProtocol.")
    if type(gaussians_a) is not type(gaussians_b):
        raise TypeError("Gaussian dataclasses must be of the same concrete type.")

    if gaussians_a.means.shape[0] != gaussians_b.means.shape[0]:
        raise ValueError("Gaussian batches must share the same batch dimension to be merged.")

    merged_kwargs = {}
    for attr in ("means", "scales", "rotations", "harmonics", "opacities"):
        merged_kwargs[attr] = torch.cat([getattr(gaussians_a, attr), getattr(gaussians_b, attr)], dim=1)
    merged_kwargs["covariances"] = build_covariance_from_scales_quaternion(
        merged_kwargs["scales"], merged_kwargs["rotations"]
    )

    return replace(gaussians_a, **merged_kwargs)
