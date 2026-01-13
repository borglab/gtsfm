"""Utility functions for transporting geometry between coordinate frames.

Authors: Ayush Baid, John Lambert, Frank Dellaert
"""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from gtsam import Point3, Pose3, Rot3, SfmTrack, Similarity3  # type: ignore

from gtsfm.common.types import CAMERA_TYPE, create_camera
from gtsfm.utils.splat import GaussiansProtocol, build_covariance_from_scales_quaternion


def Rot3s_with_so3(aRb: Rot3, rotations_b: Sequence[Rot3]) -> list[Rot3]:
    """Transport a list of Rot3s from frame ``b`` to frame ``a`` using an SO(3) transform."""
    return [aRb.compose(rotation_b) for rotation_b in rotations_b]


def optional_Rot3s_with_so3(aRb: Rot3, rotations_b: Sequence[Optional[Rot3]]) -> list[Optional[Rot3]]:
    """Transport an optional rotation list from frame ``b`` to frame ``a`` using an SO(3) transform."""
    return [aRb.compose(rotation_b) if rotation_b is not None else None for rotation_b in rotations_b]


def Pose3_map_with_se3(aTb: Pose3, pose_map_b: Mapping[int, Pose3]) -> dict[int, Pose3]:
    """Transport a Pose3 dictionary from frame ``b`` to frame ``a`` using an SE(3) transform."""
    return {i: aTb.compose(pose_b) for i, pose_b in pose_map_b.items()}


def Pose3s_with_sim3(aSb: Similarity3, poses_b: Sequence[Pose3]) -> list[Pose3]:
    """Transport a list of Pose3s from frame ``b`` to frame ``a`` using a Sim(3) transform."""
    return [aSb.transformFrom(pose_b) for pose_b in poses_b]


def optional_Pose3s_with_sim3(aSb: Similarity3, poses_b: Sequence[Optional[Pose3]]) -> list[Optional[Pose3]]:
    """Transport an optional pose list from frame ``b`` to frame ``a`` using a Sim(3) transform."""
    return [aSb.transformFrom(pose_b) if pose_b is not None else None for pose_b in poses_b]


def point_cloud_with_sim3(aSb: Similarity3, points_b: np.ndarray) -> np.ndarray:
    """Transport a point cloud from frame ``b`` to frame ``a`` using a Sim(3) transform."""
    if points_b.size == 0:
        return points_b.copy()
    transformed_points = [np.asarray(aSb.transformFrom(point_b)) for point_b in points_b]
    return np.vstack(transformed_points)


def track_with_sim3(aSb: Similarity3, track_b: SfmTrack) -> SfmTrack:
    """Transport a single SfmTrack from frame ``b`` to frame ``a`` using a Sim(3) transform."""
    track_a = SfmTrack(aSb.transformFrom(track_b.point3()))
    track_a.r = track_b.r
    track_a.g = track_b.g
    track_a.b = track_b.b
    for k in range(track_b.numberMeasurements()):
        i, uv = track_b.measurement(k)
        track_a.addMeasurement(i, uv)
    return track_a


def tracks_with_sim3(aSb: Similarity3, tracks_b: Sequence[SfmTrack]) -> list[SfmTrack]:
    """Transport a collection of SfmTracks from frame ``b`` to frame ``a`` using a Sim(3) transform."""
    return [track_with_sim3(aSb, track_b) for track_b in tracks_b]


def camera_map_with_sim3(
    aSb: Similarity3, cameras_b: Mapping[int, CAMERA_TYPE | None]
) -> dict[int, CAMERA_TYPE | None]:
    """Transport a camera dictionary from frame ``b`` to frame ``a`` using a Sim(3) transform.

    Args:
        aSb: The transformation from frame ``b`` to frame ``a``.
        cameras_b: The camera dictionary in frame ``b``, can contain None values.

    Returns:
        The camera dictionary in frame ``a``, preserving None values when present in the input.
    """
    cameras_a: dict[int, CAMERA_TYPE | None] = {}
    for i, camera_b in cameras_b.items():
        if camera_b is None:
            cameras_a[i] = None
            continue
        new_pose = aSb.transformFrom(camera_b.pose())
        cameras_a[i] = create_camera(new_pose, camera_b.calibration())
    return cameras_a


def transform_gaussian(gaussianA: dict, bSa: Similarity3) -> dict:
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

    q = Rot3.Quaternion(w, x, y, z)
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
    gaussian_splats: dict[str, torch.Tensor] | GaussiansProtocol, bSa: Similarity3
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

    converted_scales = gaussian_splats.scales * s
    converted_covariances = build_covariance_from_scales_quaternion(converted_scales, converted_rotations_xyzw)

    if not is_dataclass(gaussian_splats):
        raise TypeError("Expected gaussian_splats to be a dataclass implementing GaussiansProtocol.")

    replace_kwargs = {
        "means": converted_means,
        "scales": converted_scales,
        "rotations": converted_rotations_xyzw,
        "covariances": converted_covariances,
    }
    converted_gaussian_splats = replace(gaussian_splats, **replace_kwargs)
    return converted_gaussian_splats


def transform_camera_pose(aTc: torch.Tensor, bSa: Similarity3) -> torch.Tensor:
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

    pose_a = Pose3(Rot3(aRc), Point3(atc))

    pose_b = bSa.transformFrom(pose_a).matrix()
    bTc = torch.from_numpy(pose_b).to(aTc.device, dtype=aTc.dtype)
    return bTc
