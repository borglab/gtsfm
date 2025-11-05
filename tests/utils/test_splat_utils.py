"""Unit test on utility for 3D Gaussian Splatting util functions.

Authors: Harneet Singh Khanuja
"""

import math
import unittest
from dataclasses import dataclass

import gtsam
import numpy as np
import torch

from gtsfm.utils import splat


def dict_allclose(dict_a, dict_b, atol=1e-7):
    """Checks the all close relationship for two dictionaries"""
    if dict_a.keys() != dict_b.keys():
        return False
    return all(torch.allclose(dict_a[k], dict_b[k], atol=atol) for k in dict_a)


@dataclass
class DummyGaussians:
    """Minimal GaussiansProtocol implementation for unit testing."""

    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor
    covariances: torch.Tensor


class TestIoUtils(unittest.TestCase):
    """
    Class to test util functions for Gaussian Splatting
    """

    def test_get_rotation_matrix_from_two_vectors(self):
        """Ensures correct rotation matrix"""

        vec1 = torch.tensor([1.0, 0.0, 0.0])
        R = splat.get_rotation_matrix_from_two_vectors(vec1, vec1)
        self.assertTrue(torch.allclose(R, torch.eye(3)))

        vec2 = torch.tensor([0.0, 1.0, 0.0])
        R = splat.get_rotation_matrix_from_two_vectors(vec1, vec2)
        rotated_vec1 = R @ vec1
        self.assertTrue(torch.allclose(rotated_vec1, vec2))

    def test_auto_orient_and_center_poses(self):
        """Ensures correct orientation and centering"""

        poses = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        poses[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
        poses[1, :3, 3] = torch.tensor([3.0, 4.0, 5.0])

        # Apply a tilt to the 'up' vector (y-axis)
        tilt_angle = torch.tensor(torch.pi / 4)
        tilt_matrix = torch.tensor(
            [
                [torch.cos(tilt_angle), 0, torch.sin(tilt_angle), 0],
                [0, 1, 0, 0],
                [-torch.sin(tilt_angle), 0, torch.cos(tilt_angle), 0],
                [0, 0, 0, 1],
            ]
        )
        poses = tilt_matrix @ poses

        new_poses_centered, _ = splat.auto_orient_and_center_poses(poses.clone())
        mean_origin = new_poses_centered[:, :3, 3].mean(dim=0)
        self.assertTrue(torch.allclose(mean_origin, torch.zeros(3), atol=1e-6))

    def test_rescale_output_resolution(self):
        """Ensures consistent rescaling"""

        K = torch.tensor([[[100.0, 0.0, 50.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]])

        K_downscaled = splat.rescale_output_resolution(K.clone(), 0.5)
        expected_K_down = torch.tensor([[[50.0, 0.0, 25.0], [0.0, 60.0, 30.0], [0.0, 0.0, 1.0]]])
        self.assertTrue(torch.allclose(K_downscaled, expected_K_down))

    def test_random_quat_tensor(self):
        """Ensures correct quaternion generation."""
        N = 100
        quaternions = splat.random_quat_tensor(N)

        norms = torch.linalg.norm(quaternions, dim=1)
        self.assertEqual(quaternions.shape, (N, 4))
        self.assertTrue(torch.allclose(norms, torch.ones(N)))

    def test_get_viewmat(self):
        """Ensures correct the camera-to-world to world-to-camera matrix conversion."""

        c2w_identity = torch.eye(4).unsqueeze(0)
        w2c_identity = splat.get_viewmat(c2w_identity)
        self.assertTrue(torch.allclose(w2c_identity, c2w_identity))

        c2w = torch.eye(4)
        c2w[:3, :3] = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        c2w[:3, 3] = torch.tensor([1.0, 2.0, 3.0])

        w2c_func = splat.get_viewmat(c2w.unsqueeze(0)).squeeze(0)
        w2c_torch = torch.linalg.inv(c2w)

        self.assertTrue(torch.allclose(w2c_func, w2c_torch))

    def test_quaternion_to_matrix_xyzw(self):
        """Ensures conversion from quaternion to rotation matrix matches expectation."""

        # Identity quaternion -> identity matrix.
        identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        identity_mat = splat.quaternion_to_matrix_xyzw(identity_quat)
        self.assertEqual(identity_mat.shape, (1, 3, 3))
        self.assertTrue(torch.allclose(identity_mat[0], torch.eye(3, dtype=torch.float64)))

        # 90-degree rotation about Z axis.
        half_angle = math.pi / 4.0
        quat_z90 = torch.tensor([[0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]], dtype=torch.float32)
        rot_mat = splat.quaternion_to_matrix_xyzw(quat_z90)[0]
        expected = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(rot_mat, expected, atol=1e-6))

        # Batched input (batch_size, num_gaussians, 4).
        batched = torch.stack([identity_quat.squeeze(0).float(), quat_z90.squeeze(0)], dim=0).unsqueeze(0)
        batched_mats = splat.quaternion_to_matrix_xyzw(batched)
        self.assertEqual(batched_mats.shape, (1, 2, 3, 3))
        self.assertTrue(torch.allclose(batched_mats[0, 0], torch.eye(3)))
        self.assertTrue(torch.allclose(batched_mats[0, 1], expected, atol=1e-6))

    def test_build_covariance_from_scales_quaternion(self):
        """Ensures covariance construction follows R S Sᵀ Rᵀ."""

        scales = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
        identity_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)

        cov_identity = splat.build_covariance_from_scales_quaternion(scales, identity_quat)
        self.assertEqual(cov_identity.shape, (1, 1, 3, 3))
        expected_diag = torch.diag(torch.tensor([1.0, 4.0, 9.0], dtype=torch.float32))
        self.assertTrue(torch.allclose(cov_identity[0, 0], expected_diag))

        # With rotation about Z axis.
        half_angle = math.pi / 4.0
        quat_z90 = torch.tensor([[[0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]]], dtype=torch.float32)
        cov_rot = splat.build_covariance_from_scales_quaternion(scales, quat_z90)
        R = splat.quaternion_to_matrix_xyzw(quat_z90)[0, 0]
        expected_rot = R @ expected_diag @ R.T
        self.assertTrue(torch.allclose(cov_rot[0, 0], expected_rot, atol=1e-6))

    @staticmethod
    def _make_dummy_gaussians(base_gaussian_splats) -> DummyGaussians:
        means = base_gaussian_splats["means"].unsqueeze(0)
        scales_linear = torch.exp(base_gaussian_splats["scales"]).unsqueeze(0)
        quats_wxyz = base_gaussian_splats["quats"]
        rotations_xyzw = torch.cat([quats_wxyz[:, 1:], quats_wxyz[:, :1]], dim=-1).unsqueeze(0)

        batch_size, num_gaussians = means.shape[0], means.shape[1]
        harmonics = torch.zeros((batch_size, num_gaussians, 3, 25), dtype=means.dtype, device=means.device)
        opacities = torch.ones((batch_size, num_gaussians), dtype=means.dtype, device=means.device)

        covariances = splat.build_covariance_from_scales_quaternion(scales_linear, rotations_xyzw)

        return DummyGaussians(
            means=means,
            scales=scales_linear,
            rotations=rotations_xyzw,
            harmonics=harmonics,
            opacities=opacities,
            covariances=covariances,
        )

    def test_merge_gaussian_splats_dict(self):
        """Ensure gaussian dictionaries merge via concatenation."""

        gaussians_a = {
            "means": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.0, 0.7071, 0.0]]),
            "scales": torch.log(torch.tensor([[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])),
        }
        gaussians_b = {
            "means": torch.tensor([[2.0, 2.0, 2.0]]),
            "quats": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            "scales": torch.log(torch.tensor([[0.4, 0.4, 0.4]])),
        }

        merged = splat.merge_gaussian_splats(gaussians_a, gaussians_b)
        self.assertIsInstance(merged, dict)
        self.assertEqual(merged["means"].shape[0], 3)
        self.assertTrue(torch.allclose(merged["means"], torch.cat([gaussians_a["means"], gaussians_b["means"]], dim=0)))
        self.assertTrue(torch.allclose(merged["quats"], torch.cat([gaussians_a["quats"], gaussians_b["quats"]], dim=0)))
        self.assertTrue(
            torch.allclose(merged["scales"], torch.cat([gaussians_a["scales"], gaussians_b["scales"]], dim=0))
        )

    def test_merge_gaussian_splats_protocol(self):
        """Ensure dataclass gaussians merge via concatenation along gaussian axis."""

        base_a = {
            "means": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.9239, 0.0, 0.3827, 0.0]]),
            "scales": torch.log(torch.tensor([[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])),
        }
        base_b = {
            "means": torch.tensor([[2.0, 2.0, 2.0]]),
            "quats": torch.tensor([[0.7071, 0.0, 0.0, 0.7071]]),
            "scales": torch.log(torch.tensor([[0.4, 0.4, 0.4]])),
        }

        gaussians_a = self._make_dummy_gaussians(base_a)
        gaussians_b = self._make_dummy_gaussians(base_b)

        merged = splat.merge_gaussian_splats(gaussians_a, gaussians_b)
        self.assertIsInstance(merged, DummyGaussians)

        expected_means = torch.cat([gaussians_a.means, gaussians_b.means], dim=1)
        expected_scales = torch.cat([gaussians_a.scales, gaussians_b.scales], dim=1)
        expected_rotations = torch.cat([gaussians_a.rotations, gaussians_b.rotations], dim=1)
        expected_harmonics = torch.cat([gaussians_a.harmonics, gaussians_b.harmonics], dim=1)
        expected_opacities = torch.cat([gaussians_a.opacities, gaussians_b.opacities], dim=1)
        expected_covariances = splat.build_covariance_from_scales_quaternion(expected_scales, expected_rotations)

        self.assertTrue(torch.allclose(merged.means, expected_means))
        self.assertTrue(torch.allclose(merged.scales, expected_scales))
        self.assertTrue(torch.allclose(merged.rotations, expected_rotations))
        self.assertTrue(torch.allclose(merged.harmonics, expected_harmonics))
        self.assertTrue(torch.allclose(merged.opacities, expected_opacities))
        self.assertTrue(torch.allclose(merged.covariances, expected_covariances))
