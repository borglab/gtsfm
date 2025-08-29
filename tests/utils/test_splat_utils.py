"""Unit test on utility for 3D Gaussian Splatting util functions.

Authors: Harneet Singh Khanuja
"""

import unittest

import torch

from gtsfm.utils.splat import (
    auto_orient_and_center_poses,
    get_rotation_matrix_from_two_vectors,
    get_viewmat,
    random_quat_tensor,
    rescale_output_resolution,
)


class TestIoUtils(unittest.TestCase):
    def test_get_rotation_matrix_from_two_vectors(self):
        """Ensures correct rotation matrix"""

        vec1 = torch.tensor([1.0, 0.0, 0.0])
        R = get_rotation_matrix_from_two_vectors(vec1, vec1)
        self.assertTrue(torch.allclose(R, torch.eye(3)))

        vec2 = torch.tensor([0.0, 1.0, 0.0])
        R = get_rotation_matrix_from_two_vectors(vec1, vec2)
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

        new_poses_centered, _ = auto_orient_and_center_poses(poses.clone())
        mean_origin = new_poses_centered[:, :3, 3].mean(dim=0)
        self.assertTrue(torch.allclose(mean_origin, torch.zeros(3), atol=1e-6))

    def test_rescale_output_resolution(self):
        """Ensures consistent rescaling"""

        K = torch.tensor([[[100.0, 0.0, 50.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]])

        K_downscaled = rescale_output_resolution(K.clone(), 0.5)
        expected_K_down = torch.tensor([[[50.0, 0.0, 25.0], [0.0, 60.0, 30.0], [0.0, 0.0, 1.0]]])
        self.assertTrue(torch.allclose(K_downscaled, expected_K_down))

    def test_random_quat_tensor(self):
        """Ensures correct quaternion generation."""
        N = 100
        quats = random_quat_tensor(N)

        norms = torch.linalg.norm(quats, dim=1)
        self.assertEqual(quats.shape, (N, 4))
        self.assertTrue(torch.allclose(norms, torch.ones(N)))

    def test_get_viewmat(self):
        """Ensures correct the camera-to-world to world-to-camera matrix conversion."""

        c2w_identity = torch.eye(4).unsqueeze(0)
        w2c_identity = get_viewmat(c2w_identity)
        self.assertTrue(torch.allclose(w2c_identity, c2w_identity))

        c2w = torch.eye(4)
        c2w[:3, :3] = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        c2w[:3, 3] = torch.tensor([1.0, 2.0, 3.0])

        w2c_func = get_viewmat(c2w.unsqueeze(0)).squeeze(0)
        w2c_torch = torch.linalg.inv(c2w)

        self.assertTrue(torch.allclose(w2c_func, w2c_torch))


if __name__ == "__main__":
    unittest.main()
