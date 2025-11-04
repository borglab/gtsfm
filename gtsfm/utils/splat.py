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


def transform_gaussian(gaussianA: dict, bSa: gtsam.Similarity3) -> dict:
    """
    Transforms a Gaussian Splat from one coordinate system to another using gtsam.Similarity3
    Args:
        gaussianA (dict): A dictionary representing the Gaussian in coordinate system A.
        bSa (gtsam.Similarity3): The transformation from coordinate system A to B.
    Returns:
        gaussianB: A dictionary representing the Gaussian in coordinate system B.
    """
    meanA = gaussianA["mean"]
    meanB = torch.Tensor(bSa.transformFrom(meanA))

    w = gaussianA["quat"][0]
    x = gaussianA["quat"][1]
    y = gaussianA["quat"][2]
    z = gaussianA["quat"][3]

    q = gtsam.Rot3.Quaternion(w, x, y, z)
    bRa = bSa.rotation()
    rotationB = torch.Tensor((bRa * q).toQuaternion().coeffs())[[3, 0, 1, 2]]

    if rotationB[0] < 0:
        rotationB *= -1.0

    scaleB = torch.log(torch.tensor(bSa.scale())) + gaussianA["scale"]

    # we only update the means, quaternions and scales (covariance) as opacity and color do not change.
    gaussianB = gaussianA.copy()
    gaussianB["mean"] = meanB
    gaussianB["quat"] = rotationB
    gaussianB["scale"] = scaleB

    return gaussianB


def transform_gaussian_splats(
    gaussian_splats: dict[str, torch.Tensor] | GaussiansProtocol, bSa: gtsam.Similarity3
) -> dict[str, torch.Tensor] | GaussiansProtocol:
    """
    Transforms Gaussian splats from one coordinate system to another using ``gtsam.Similarity3``.

    Args:
        gaussian_splats: Collection of Gaussians expressed either as a dictionary containing
            ``means`` (N,3), ``quats`` (N,4) in ``w,x,y,z`` order, and ``scales`` (N,3) stored in
            log-space; or an object adhering to :class:`GaussiansProtocol` with rotations in ``x,y,z,w`` order
            and scales expressed linearly and shape having batch_size = 1 in dim 0.
        bSa: The transformation from coordinate system ``A`` to ``B``.

    Returns:
        Gaussian splats in coordinate system ``B`` with the same container type as the input.
    """

    def _apply_quaternion_left_multiply(sim3_quat_wxyz: torch.Tensor, quats_wxyz: torch.Tensor) -> torch.Tensor:
        """Left-multiply ``quats_wxyz`` by ``sim3_quat_wxyz`` (both in w,x,y,z order)."""
        w0, x0, y0, z0 = sim3_quat_wxyz
        w1, x1, y1, z1 = quats_wxyz.unbind(-1)
        converted_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        converted_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        converted_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        converted_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
        converted = torch.stack([converted_w, converted_x, converted_y, converted_z], dim=-1)
        converted = converted / converted.norm(dim=-1, keepdim=True)
        converted[converted[..., 0] < 0] *= -1.0
        return converted

    is_dict_input = isinstance(gaussian_splats, dict)
    means = gaussian_splats["means"] if is_dict_input else gaussian_splats.means
    dtype = means.dtype
    device = means.device

    R_np = bSa.rotation().matrix()
    t_np = bSa.translation()
    scale_value = float(bSa.scale())

    R = torch.tensor(R_np, dtype=dtype, device=device)
    t = torch.tensor(t_np, dtype=dtype, device=device)
    s = torch.tensor(scale_value, dtype=dtype, device=device)

    if means.ndim == 2:
        converted_means = torch.matmul(means, R.T) + t
    else:
        converted_means = torch.matmul(means, R.T) + t.unsqueeze(0)
    converted_means = converted_means * s

    sim3_quat_coeffs = torch.tensor(bSa.rotation().toQuaternion().coeffs(), dtype=dtype, device=device)
    sim3_quat_wxyz = sim3_quat_coeffs[[3, 0, 1, 2]]

    if is_dict_input:
        quats_wxyz = gaussian_splats["quats"]
        converted_quaternions = _apply_quaternion_left_multiply(
            sim3_quat_wxyz.to(dtype=quats_wxyz.dtype, device=quats_wxyz.device),
            quats_wxyz,
        )
        scales_tensor = gaussian_splats["scales"]
        scale_offset = torch.log(torch.tensor(scale_value, dtype=scales_tensor.dtype, device=scales_tensor.device))
        converted_scales = scales_tensor + scale_offset

        converted_gaussian_splats = gaussian_splats.copy()
        converted_gaussian_splats["means"] = converted_means
        converted_gaussian_splats["quats"] = converted_quaternions
        converted_gaussian_splats["scales"] = converted_scales
        return converted_gaussian_splats

    rotations_xyzw = gaussian_splats.rotations
    rotations_wxyz = torch.cat([rotations_xyzw[..., 3:], rotations_xyzw[..., :3]], dim=-1)
    converted_rotations_wxyz = _apply_quaternion_left_multiply(
        sim3_quat_wxyz.to(dtype=rotations_xyzw.dtype, device=rotations_xyzw.device),
        rotations_wxyz,
    )
    converted_rotations_xyzw = torch.cat([converted_rotations_wxyz[..., 1:], converted_rotations_wxyz[..., :1]], dim=-1)

    converted_scales = gaussian_splats.scales * torch.tensor(
        scale_value, dtype=gaussian_splats.scales.dtype, device=gaussian_splats.scales.device
    )

    if not is_dataclass(gaussian_splats):
        raise TypeError("Expected gaussian_splats to be a dataclass implementing GaussiansProtocol.")

    replace_kwargs = {
        "means": converted_means,
        "scales": converted_scales,
        "rotations": converted_rotations_xyzw,
    }
    converted_gaussian_splats = replace(gaussian_splats, **replace_kwargs)
    return converted_gaussian_splats


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
    for attr in ("means", "scales", "rotations", "harmonics", "opacities", "covariances"):
        merged_kwargs[attr] = torch.cat([getattr(gaussians_a, attr), getattr(gaussians_b, attr)], dim=1)

    return replace(gaussians_a, **merged_kwargs)


def transform_camera_pose(aTc: torch.Tensor, bSa: gtsam.Similarity3) -> torch.Tensor:
    """
    Transforms camera pose from coordinate frame A to B
    Args:
        aTc: 4x4 camera-to-world pose in frame A
        bSa (gtsam.Similarity3): The transformation from coordinate system A to B.

    Returns:
        bTc: 4x4 camera-to-world pose in frame B
    """
    assert aTc.shape == (4, 4)

    aRc = aTc[:3, :3].cpu().numpy()
    atc = aTc[:3, 3:4].cpu().numpy().squeeze()

    pose_a = gtsam.Pose3(gtsam.Rot3(aRc), gtsam.Point3(atc))

    pose_b = bSa.transformFrom(pose_a).matrix()
    bTc = torch.from_numpy(pose_b).to(aTc.device, dtype=aTc.dtype)
    return bTc
