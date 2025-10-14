"""Unit test on utility for 3D Gaussian Splatting util functions.

Authors: Harneet Singh Khanuja
"""

import unittest

import gtsam
import numpy as np
import torch

from gtsfm.utils import splat


def dict_allclose(dict_a, dict_b, atol=1e-7):
    """Checks the all close relationship for two dictionaries"""
    if dict_a.keys() != dict_b.keys():
        return False
    return all(torch.allclose(dict_a[k], dict_b[k], atol=atol) for k in dict_a)


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

    def test_transform_gaussian(self):
        """Ensures correct transformation from coordinate system A to B"""

        base_gaussian = {
            "mean": torch.Tensor([0.5, 0, 0]),
            "quat": torch.Tensor([0.92387953, 0, 0.38268343, 0]),
            "scale": torch.log(torch.Tensor([0.3, 0.15, 0.05])),
        }

        # --- Test 1: Translation Only ---
        rot1 = gtsam.Rot3()
        trans1 = gtsam.Point3(1.5, 1, 0.5)
        scale1 = 1.0
        sim3_1 = gtsam.Similarity3(rot1, trans1, scale1)
        gauss_B_1 = splat.transform_gaussian(base_gaussian, sim3_1)
        expected_mean_1 = base_gaussian["mean"] + trans1
        self.assertTrue(torch.allclose(gauss_B_1["mean"], expected_mean_1.to(torch.float32)))

        # --- Test 2: Rotation Only ---
        rot2 = gtsam.Rot3.Rz(np.deg2rad(90))
        trans2 = gtsam.Point3(0, 0, 0)
        scale2 = 1.0
        sim3_2 = gtsam.Similarity3(rot2, trans2, scale2)
        gauss_B_2 = splat.transform_gaussian(base_gaussian, sim3_2)
        expected_mean_2 = torch.Tensor(rot2.matrix()) @ base_gaussian["mean"]
        self.assertTrue(torch.allclose(gauss_B_2["mean"], expected_mean_2))

        # --- Test 3: Scale Only ---
        rot3 = gtsam.Rot3()
        trans3 = gtsam.Point3(0, 0, 0)
        scale3 = torch.tensor(2.5)
        sim3_3 = gtsam.Similarity3(rot3, trans3, scale3)
        gauss_B_3 = splat.transform_gaussian(base_gaussian, sim3_3)
        expected_mean_3 = scale3 * base_gaussian["mean"]
        expected_scale_3 = torch.log(scale3) + base_gaussian["scale"]
        self.assertTrue(torch.allclose(gauss_B_3["mean"], expected_mean_3))
        self.assertTrue(torch.allclose(gauss_B_3["scale"], expected_scale_3))

        # --- Test 4: Combined Transformation ---
        rot4 = gtsam.Rot3(0.82236317, 0.02226003, 0.43967974, 0.36042341)
        R_4 = rot4.matrix()
        trans4 = gtsam.Point3(-1, 0.5, 1)
        scale4 = torch.tensor(0.5)
        sim3_4 = gtsam.Similarity3(rot4, trans4, scale4)
        gauss_B_4 = splat.transform_gaussian(base_gaussian, sim3_4)
        expected_mean_4 = scale4 * (torch.Tensor(R_4) @ base_gaussian["mean"] + trans4)
        expected_scale_4 = torch.log(scale4) + base_gaussian["scale"]
        base_gaussian_quaternion = gtsam.Rot3(
            base_gaussian["quat"][0],
            base_gaussian["quat"][1],
            base_gaussian["quat"][2],
            base_gaussian["quat"][3],
        )
        expected_rotation_4 = R_4 @ base_gaussian_quaternion.matrix()
        self.assertTrue(torch.allclose(gauss_B_4["mean"], expected_mean_4.to(torch.float32)))
        self.assertTrue(torch.allclose(gauss_B_4["scale"], expected_scale_4))
        self.assertTrue(
            torch.allclose(
                gauss_B_4["quat"], torch.Tensor(gtsam.Rot3(expected_rotation_4).toQuaternion().coeffs()[[3, 0, 1, 2]])
            )
        )

    def test_transform_gaussian_splats(self):
        """Ensures correct transformation from coordinate system A to B"""

        base_gaussian_splats = {
            "means": torch.Tensor([[0.5, 0, 0], [2.5, 0, 0], [0, 1.4, 3]]),
            "quats": torch.Tensor(
                [[0.92387953, 0, 0.38268343, 0], [0.70710678, 0, 0, 0.70710678], [0.5, -0.5, 0.5, 0.5]]
            ),
            "scales": torch.log(torch.Tensor([[0.3, 0.15, 0.05], [0.4, 2.3, 1.5], [0.1, 0.2, 1]])),
        }

        test_gaussian_splats = {"means": [], "quats": [], "scales": []}

        for i in range(len(base_gaussian_splats["means"])):
            base_gaussian = {
                "mean": base_gaussian_splats["means"][i],
                "quat": base_gaussian_splats["quats"][i],
                "scale": base_gaussian_splats["scales"][i],
            }

            # --- Test 1: Translation Only ---
            sim3_1 = gtsam.Similarity3(gtsam.Rot3(), gtsam.Point3(1.5, 1, 0.5), torch.tensor(1.0))
            gauss_B_1 = splat.transform_gaussian(base_gaussian, sim3_1)

            test_gaussian_splats["means"].append(gauss_B_1["mean"])
            test_gaussian_splats["quats"].append(gauss_B_1["quat"])
            test_gaussian_splats["scales"].append(gauss_B_1["scale"])

        for key in test_gaussian_splats:
            test_gaussian_splats[key] = torch.stack(test_gaussian_splats[key])

        self.assertTrue(
            dict_allclose(test_gaussian_splats, splat.transform_gaussian_splats(base_gaussian_splats, sim3_1))
        )

        test_gaussian_splats = {"means": [], "quats": [], "scales": []}

        for i in range(len(base_gaussian_splats["means"])):
            base_gaussian = {
                "mean": base_gaussian_splats["means"][i],
                "quat": base_gaussian_splats["quats"][i],
                "scale": base_gaussian_splats["scales"][i],
            }

            # --- Test 2: Rotation Only ---
            sim3_2 = gtsam.Similarity3(gtsam.Rot3.Rz(np.deg2rad(90)), gtsam.Point3(0, 0, 0), torch.tensor(1.0))
            gauss_B_2 = splat.transform_gaussian(base_gaussian, sim3_2)

            test_gaussian_splats["means"].append(gauss_B_2["mean"])
            test_gaussian_splats["quats"].append(gauss_B_2["quat"])
            test_gaussian_splats["scales"].append(gauss_B_2["scale"])

        for key in test_gaussian_splats:
            test_gaussian_splats[key] = torch.stack(test_gaussian_splats[key])

        self.assertTrue(
            dict_allclose(test_gaussian_splats, splat.transform_gaussian_splats(base_gaussian_splats, sim3_2))
        )

        test_gaussian_splats = {"means": [], "quats": [], "scales": []}

        for i in range(len(base_gaussian_splats["means"])):
            base_gaussian = {
                "mean": base_gaussian_splats["means"][i],
                "quat": base_gaussian_splats["quats"][i],
                "scale": base_gaussian_splats["scales"][i],
            }

            # --- Test 3: Scale Only ---
            sim3_3 = gtsam.Similarity3(gtsam.Rot3(), gtsam.Point3(0, 0, 0), torch.tensor(2.5))
            gauss_B_3 = splat.transform_gaussian(base_gaussian, sim3_3)

            test_gaussian_splats["means"].append(gauss_B_3["mean"])
            test_gaussian_splats["quats"].append(gauss_B_3["quat"])
            test_gaussian_splats["scales"].append(gauss_B_3["scale"])

        for key in test_gaussian_splats:
            test_gaussian_splats[key] = torch.stack(test_gaussian_splats[key])

        self.assertTrue(
            dict_allclose(test_gaussian_splats, splat.transform_gaussian_splats(base_gaussian_splats, sim3_3))
        )

        test_gaussian_splats = {"means": [], "quats": [], "scales": []}

        for i in range(len(base_gaussian_splats["means"])):
            base_gaussian = {
                "mean": base_gaussian_splats["means"][i],
                "quat": base_gaussian_splats["quats"][i],
                "scale": base_gaussian_splats["scales"][i],
            }

            # --- Test 4: Combined Transformation ---
            sim3_4 = gtsam.Similarity3(
                gtsam.Rot3(0.82236317, 0.02226003, 0.43967974, 0.36042341), gtsam.Point3(-1, 0.5, 1), torch.tensor(0.5)
            )
            gauss_B_4 = splat.transform_gaussian(base_gaussian, sim3_4)

            test_gaussian_splats["means"].append(gauss_B_4["mean"])
            test_gaussian_splats["quats"].append(gauss_B_4["quat"])
            test_gaussian_splats["scales"].append(gauss_B_4["scale"])

        for key in test_gaussian_splats:
            test_gaussian_splats[key] = torch.stack(test_gaussian_splats[key])

        self.assertTrue(
            dict_allclose(test_gaussian_splats, splat.transform_gaussian_splats(base_gaussian_splats, sim3_4))
        )

    def test_transform_camera_poses(self):
        """Ensures correct transformation from coordinate system A to B"""

        # --- Test 1: Identity transform ---
        pose_a = torch.eye(4)
        bSa = gtsam.Similarity3()
        pose_b = splat.transform_camera_pose(pose_a, bSa)
        self.assertTrue(torch.allclose(pose_a, pose_b))

        # --- Test 2: Translation only ---
        pose_a = torch.eye(4)
        pose_a[0, 3] = 1.0
        pose_a[1, 3] = 2.4

        # Translate frame A by (2, 3, 4) to get frame B
        bSa = gtsam.Similarity3(gtsam.Rot3(), gtsam.Point3(2, 3, 4), 1.0)
        pose_b = splat.transform_camera_pose(pose_a, bSa)

        expected_pose_b = torch.eye(4)
        expected_pose_b[:3, 3] = torch.tensor([3.0, 5.4, 4.0])
        self.assertTrue(torch.allclose(pose_b, expected_pose_b))

        # --- Test 3: Rotation only ---
        pose_a = torch.eye(4)
        pose_a[0, 3] = 1.0  # Camera at (1, 0, 0)
        print(pose_a)

        # Rotate frame A by 90 degrees around Z axis to get frame B
        bSa = gtsam.Similarity3(gtsam.Rot3.Rz(np.pi / 2), gtsam.Point3(0, 0, 0), 1.0)

        pose_b = splat.transform_camera_pose(pose_a, bSa)

        expected_pose_b = torch.eye(4)
        # Original position (1,0,0) rotates to (0,1,0)
        expected_pose_b[1, 3] = 1.0
        # Original orientation (I) rotates by Rz(90)
        expected_pose_b[:3, :3] = pose_a[:3, :3] @ bSa.rotation().matrix()
        self.assertTrue(torch.allclose(pose_b, expected_pose_b))

        # --- Test 4: Scale only ---
        pose_a = torch.eye(4)
        pose_a[0, 3] = 2.0
        pose_a[1, 3] = 3.4
        pose_a[2, 3] = 1.2  # Camera at (2, 3.4, 1.2)

        # Scale frame A by a factor of 3 to get frame B
        bSa = gtsam.Similarity3(gtsam.Rot3(), gtsam.Point3(0, 0, 0), 3.0)

        pose_b = splat.transform_camera_pose(pose_a, bSa)

        expected_pose_b = torch.eye(4)
        expected_pose_b[:3, 3] = torch.tensor([6.0, 10.2, 3.6])
        self.assertTrue(torch.allclose(pose_b, expected_pose_b))

        # --- Test 4: Combined Transformation ---
        pose_a = torch.eye(4)
        pose_a[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        pose_a[:3, 3] = torch.tensor([1.0, 0.0, 0.0])

        # Transformation from A to B:
        # - Rotate -90 deg around Z
        # - Translate by (1, 2, 3)
        # - Scale by 2
        bSa = gtsam.Similarity3(gtsam.Rot3.Rz(-np.pi / 2), gtsam.Point3(1, 2, 3), 2.0)

        pose_b = splat.transform_camera_pose(pose_a, bSa)

        # Expected result calculation:
        # bRc = bRa @ aRc = Rz(-90) @ Rz(90) = Identity
        # btc = s * (bRa @ atc + btA)
        #      = 2 * ( Rz(-90) @ (1,0,0) + (1,2,3) )
        #      = 2 * ( (0,-1,0) + (1,2,3) )
        #      = 2 * (1,1,3) = (2,2,6)
        expected_pose_b = torch.eye(4)
        expected_pose_b[:3, 3] = torch.tensor([2.0, 2.0, 6.0])
        self.assertTrue(torch.allclose(pose_b, expected_pose_b))


if __name__ == "__main__":
    unittest.main()
