"""Unit tests for rotation utils.

Authors: Ayush Baid
"""
import unittest
from typing import Dict, List, Tuple

import numpy as np
from gtsam import Rot3

import gtsfm.utils.geometry_comparisons as geometry_comparisons
import gtsfm.utils.rotation as rotation_util
import tests.data.sample_poses as sample_poses

ROTATION_ANGLE_ERROR_THRESHOLD_DEG = 2


RELATIVE_ROTATION_DICT = Dict[Tuple[int, int], Rot3]


def _get_ordered_chain_pose_data() -> Tuple[RELATIVE_ROTATION_DICT, List[float]]:
    """Return data for a scenario with 5 camera poses, with ordering that follows their connectivity.

    Accordingly, we specify i1 < i2 for all edges (i1,i2).

    Graph topology:

              | 2     | 3
              o-- ... o--
              .       .
              .       .
    |         |       |
    o-- ... --o     --o
    0         1       4

    Returns:
        Tuple of mapping from image index pair to relative rotations, and expected global rotation angles.
    """
    # Expected angles.
    wRi_list_euler_deg_expected = np.array([0, 90, 0, 0, 90])

    # Ground truth 3d rotations for 5 ordered poses (0,1,2,3,4)
    wRi_list_gt = [Rot3.RzRyRx(np.deg2rad(Rz_deg), 0, 0) for Rz_deg in wRi_list_euler_deg_expected]

    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    i2Ri1_dict = _create_synthetic_relative_pose_measurements(wRi_list_gt, edges=edges)

    return i2Ri1_dict, wRi_list_euler_deg_expected


def _get_mixed_order_chain_pose_data() -> Tuple[RELATIVE_ROTATION_DICT, List[float]]:
    """Return data for a scenario with 5 camera poses, with ordering that does NOT follow their connectivity.

    Below, we do NOT specify i1 < i2 for all edges (i1,i2).

    Graph topology:

              | 3     | 0
              o-- ... o--
              .       .
              .       .
    |         |       |
    o-- ... --o     --o
    4         1       2

    """
    # Expected angles.
    wRi_list_euler_deg_expected = np.array([0, 90, 90, 0, 0])

    # Ground truth 2d rotations for 5 ordered poses (0,1,2,3,4)
    wRi_list_gt = [Rot3.RzRyRx(np.deg2rad(Rz_deg), 0, 0) for Rz_deg in wRi_list_euler_deg_expected]

    edges = [(1, 4), (1, 3), (0, 3), (0, 2)]
    i2Ri1_dict = _create_synthetic_relative_pose_measurements(wRi_list_gt=wRi_list_gt, edges=edges)

    return i2Ri1_dict, wRi_list_euler_deg_expected


def _create_synthetic_relative_pose_measurements(
    wRi_list_gt: List[Rot3], edges: List[Tuple[int, int]]
) -> Dict[Tuple[int, int], Rot3]:
    """Generate synthetic relative rotation measurements, from ground truth global rotations.

    Args:
        wRi_list_gt: List of (3,3) rotation matrices.
        edges: Edges as pairs of image indices.

    Returns:
        Relative rotation measurements.
    """
    i2Ri1_dict = {}
    for i1, i2 in edges:
        wRi2 = wRi_list_gt[i2]
        wRi1 = wRi_list_gt[i1]
        i2Ri1_dict[(i1, i2)] = wRi2.inverse() * wRi1

    return i2Ri1_dict


def _wrap_angles(angles: np.ndarray) -> np.ndarray:
    """Map angle (in degrees) from domain [-\infty, \infty] to [0, 360).

    Args:
        angles: Array of shape (N,) representing angles (in degrees) in any interval.

    Returns:
        Array of shape (N,) representing the angles (in degrees) mapped to the interval [0, 360].
    """
    # Reduce the angle
    angles = angles % 360

    # Force it to be the positive remainder, so that 0 <= angle < 360
    angles = (angles + 360) % 360
    return angles


class TestRotationUtil(unittest.TestCase):
    def test_mst_initialization(self):
        """Test for 4 poses in a circle, with a pose connected all others."""
        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.CIRCLE_ALL_EDGES_GLOBAL_POSES, sample_poses.CIRCLE_ALL_EDGES_RELATIVE_POSES
        )

        wRi_computed = rotation_util.initialize_global_rotations_using_mst(
            len(wRi_expected),
            i2Ri1_dict,
            edge_weights={(i1, i2): (i1 + i2) * 100 for i1, i2 in i2Ri1_dict.keys()},
        )
        self.assertTrue(
            geometry_comparisons.compare_rotations(wRi_computed, wRi_expected, ROTATION_ANGLE_ERROR_THRESHOLD_DEG)
        )

    def test_greedily_construct_st_ordered_chain(self) -> None:
        """Ensures that we can greedily construct a Spanning Tree for an ordered chain."""

        i2Ri1_dict, wRi_list_euler_deg_expected = _get_ordered_chain_pose_data()

        num_images = 5
        wRi_list_computed = rotation_util.initialize_global_rotations_using_mst(
            num_images,
            i2Ri1_dict,
            edge_weights={(i1, i2): (i1 + i2) * 100 for i1, i2 in i2Ri1_dict.keys()},
        )

        wRi_list_euler_deg_est = [np.rad2deg(wRi.roll()) for wRi in wRi_list_computed]
        assert np.allclose(wRi_list_euler_deg_est, wRi_list_euler_deg_expected)

    def test_greedily_construct_st_mixed_order_chain(self) -> None:
        """Ensures that we can greedily construct a Spanning Tree for an unordered chain."""
        i2Ri1_dict, wRi_list_euler_deg_expected = _get_mixed_order_chain_pose_data()

        num_images = 5
        wRi_list_computed = rotation_util.initialize_global_rotations_using_mst(
            num_images,
            i2Ri1_dict,
            edge_weights={(i1, i2): (i1 + i2) * 100 for i1, i2 in i2Ri1_dict.keys()},
        )

        wRi_list_euler_deg_est = np.array([np.rad2deg(wRi.roll()) for wRi in wRi_list_computed])

        # Make sure both lists of angles start at 0 deg.
        wRi_list_euler_deg_est -= wRi_list_euler_deg_est[0]
        wRi_list_euler_deg_expected -= wRi_list_euler_deg_expected[0]

        assert np.allclose(_wrap_angles(wRi_list_euler_deg_est), _wrap_angles(wRi_list_euler_deg_expected))


if __name__ == "__main__":
    unittest.main()
