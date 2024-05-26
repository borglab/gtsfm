"""Tests for Shonan rotation averaging.

Authors: Ayush Baid, John Lambert
"""

import pickle
import random
import unittest
from typing import Dict, List, Tuple
from pathlib import Path

import dask
import numpy as np
from gtsam import Pose3, Rot3

import gtsfm.utils.geometry_comparisons as geometry_comparisons
import gtsfm.utils.io as io_utils
import gtsfm.utils.rotation as rotation_util
import tests.data.sample_poses as sample_poses
from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging
from gtsfm.common.pose_prior import PosePrior, PosePriorType

ROTATION_ANGLE_ERROR_THRESHOLD_DEG = 2
TEST_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
LARGE_PROBLEM_BAL_FILE = TEST_DATA_ROOT / "problem-394-100368-pre.txt"


class TestShonanRotationAveraging(unittest.TestCase):
    """Test class for Shonan rotation averaging.

    All unit test functions defined in TestRotationAveragingBase are run automatically.
    """

    def setUp(self) -> None:
        super().setUp()

        self.obj = ShonanRotationAveraging()

    def __execute_test(self, i2Ri1_input: Dict[Tuple[int, int], Rot3], wRi_expected: List[Rot3]) -> None:
        """Helper function to run the averagaing and assert w/ expected.

        Args:
            i2Ri1_input: Relative rotations, which are input to the algorithm.
            wRi_expected: Expected global rotations.
        """
        i1Ti2_priors: Dict[Tuple[int, int], PosePrior] = {}
        v_corr_idxs = {(i1, i2): _generate_corr_idxs(random.randint(0, 100)) for i1, i2 in i2Ri1_input.keys()}
        wRi_computed = self.obj.run_rotation_averaging(len(wRi_expected), i2Ri1_input, i1Ti2_priors, v_corr_idxs)
        self.assertTrue(
            geometry_comparisons.compare_rotations(wRi_computed, wRi_expected, ROTATION_ANGLE_ERROR_THRESHOLD_DEG)
        )

    def test_circle_two_edges(self):
        """Test for 4 poses in a circle, with a pose connected to its immediate neighborhood."""
        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.CIRCLE_TWO_EDGES_GLOBAL_POSES, sample_poses.CIRCLE_TWO_EDGES_RELATIVE_POSES
        )
        self.__execute_test(i2Ri1_dict, wRi_expected)

    def test_circle_all_edges(self):
        """Test for 4 poses in a circle, with a pose connected all others."""
        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.CIRCLE_ALL_EDGES_GLOBAL_POSES, sample_poses.CIRCLE_ALL_EDGES_RELATIVE_POSES
        )
        self.__execute_test(i2Ri1_dict, wRi_expected)

    def test_line_large_edges(self):
        """Test for 3 poses in a line, with large translations between them."""
        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.LINE_LARGE_EDGES_GLOBAL_POSES, sample_poses.LINE_LARGE_EDGES_RELATIVE_POSES
        )
        self.__execute_test(i2Ri1_dict, wRi_expected)

    def test_panorama(self):
        """Test for 3 poses in a panorama configuration (large rotations at the same location)"""
        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.PANORAMA_GLOBAL_POSES, sample_poses.PANORAMA_RELATIVE_POSES
        )
        self.__execute_test(i2Ri1_dict, wRi_expected)

    def test_simple_three_nodes_two_measurements(self):
        """Test a simple case with three relative rotations."""

        i0Ri1 = Rot3.RzRyRx(0, np.deg2rad(30), 0)
        i1Ri2 = Rot3.RzRyRx(0, 0, np.deg2rad(20))
        i0Ri2 = i0Ri1.compose(i1Ri2)

        i2Ri1_dict = {(0, 1): i0Ri1.inverse(), (1, 2): i1Ri2.inverse()}

        expected_wRi_list = [Rot3(), i0Ri1, i0Ri2]

        self.__execute_test(i2Ri1_dict, expected_wRi_list)

    def test_simple_with_prior(self):
        """Test a simple case with 1 measurement and a single pose prior."""
        expected_wRi_list = [Rot3.RzRyRx(0, 0, 0), Rot3.RzRyRx(0, np.deg2rad(30), 0), Rot3.RzRyRx(np.deg2rad(30), 0, 0)]

        i2Ri1_dict = {(0, 1): expected_wRi_list[1].between(expected_wRi_list[0])}

        expected_0R2 = expected_wRi_list[0].between(expected_wRi_list[2])
        i1Ti2_priors = {
            (0, 2): PosePrior(
                value=Pose3(expected_0R2, np.zeros((3,))),
                covariance=np.eye(6) * 1e-5,
                type=PosePriorType.SOFT_CONSTRAINT,
            )
        }

        v_corr_idxs = {
            (0, 1): _generate_corr_idxs(1),
            (0, 2): _generate_corr_idxs(1),
        }

        wRi_computed = self.obj.run_rotation_averaging(len(expected_wRi_list), i2Ri1_dict, i1Ti2_priors, v_corr_idxs)
        self.assertTrue(
            geometry_comparisons.compare_rotations(wRi_computed, expected_wRi_list, ROTATION_ANGLE_ERROR_THRESHOLD_DEG)
        )

    def test_computation_graph(self):
        """Test the dask computation graph execution using a valid collection of relative poses."""

        num_poses = 3

        i2Ri1_dict = {
            (0, 1): Rot3.RzRyRx(0, np.deg2rad(30), 0),
            (1, 2): Rot3.RzRyRx(0, 0, np.deg2rad(20)),
        }
        v_corr_idxs = {
            (0, 1): _generate_corr_idxs(200),
            (1, 2): _generate_corr_idxs(500),
        }

        i2Ri1_graph = dask.delayed(i2Ri1_dict)

        # use the GTSAM API directly (without dask) for rotation averaging
        i1Ti2_priors: Dict[Tuple[int, int], PosePrior] = {}
        expected_wRi_list = self.obj.run_rotation_averaging(num_poses, i2Ri1_dict, i1Ti2_priors, v_corr_idxs)

        # use dask's computation graph
        gt_wTi_list = [None] * len(expected_wRi_list)
        rotations_graph, _ = self.obj.create_computation_graph(
            num_poses, i2Ri1_graph, i1Ti2_priors, gt_wTi_list, v_corr_idxs
        )

        with dask.config.set(scheduler="single-threaded"):
            wRi_list = dask.compute(rotations_graph)[0]

        # compare the two results
        self.assertTrue(
            geometry_comparisons.compare_rotations(wRi_list, expected_wRi_list, ROTATION_ANGLE_ERROR_THRESHOLD_DEG)
        )

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")

    def test_nonconsecutive_indices(self):
        """Run rotation averaging on a graph with indices that are not ordered as [0,...,N-1].

        Note: Test would fail if Shonan keys were not temporarily re-ordered inside the implementation.
        See https://github.com/borglab/gtsam/issues/784
        """
        num_images = 4

        # Assume pose 0 is orphaned in the visibility graph
        # Let wTi0's (R,t) be parameterized as identity Rot3(), and t = [1,1,0]
        wTi1 = Pose3(Rot3(), np.array([3, 1, 0]))
        wTi2 = Pose3(Rot3(), np.array([3, 3, 0]))
        wTi3 = Pose3(Rot3(), np.array([1, 3, 0]))

        # generate i2Ri1 rotations
        # (i1,i2) -> i2Ri1
        i2Ri1_input = {
            (1, 2): wTi2.between(wTi1).rotation(),
            (2, 3): wTi3.between(wTi2).rotation(),
            (1, 3): wTi3.between(wTi1).rotation(),
        }

        # Keys do not overlap with i2Ri1_dict.
        v_corr_idxs = {
            (1, 2): _generate_corr_idxs(200),
            (1, 3): _generate_corr_idxs(500),
            (0, 2): _generate_corr_idxs(0),
        }

        relative_pose_priors: Dict[Tuple[int, int], PosePrior] = {}
        wRi_computed = self.obj.run_rotation_averaging(num_images, i2Ri1_input, relative_pose_priors, v_corr_idxs)
        wRi_expected = [None, wTi1.rotation(), wTi2.rotation(), wTi3.rotation()]
        self.assertTrue(
            geometry_comparisons.compare_rotations(wRi_computed, wRi_expected, angular_error_threshold_degrees=0.1)
        )

    def test_initialization(self) -> None:
        """Test that the result of Shonan is not dependent on the initialization."""
        i2Ri1_dict_noisefree, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.CIRCLE_ALL_EDGES_GLOBAL_POSES, sample_poses.CIRCLE_ALL_EDGES_RELATIVE_POSES
        )
        v_corr_idxs = {pair: _generate_corr_idxs(random.randint(1, 10)) for pair in i2Ri1_dict_noisefree.keys()}

        # Add noise to the relative rotations
        i2Ri1_dict_noisy = {
            pair: i2Ri1 * rotation_util.random_rotation() for pair, i2Ri1 in i2Ri1_dict_noisefree.items()
        }

        wRi_computed_with_random_init = self.obj.run_rotation_averaging(
            num_images=len(wRi_expected),
            i2Ri1_dict=i2Ri1_dict_noisy,
            i1Ti2_priors={},
            v_corr_idxs=v_corr_idxs,
        )

        shonan_mst_init = ShonanRotationAveraging(use_mst_init=True)
        wRi_computed_with_mst_init = shonan_mst_init.run_rotation_averaging(
            num_images=len(wRi_expected),
            i2Ri1_dict=i2Ri1_dict_noisy,
            i1Ti2_priors={},
            v_corr_idxs=v_corr_idxs,
        )

        self.assertTrue(
            geometry_comparisons.compare_rotations(
                wRi_computed_with_random_init, wRi_computed_with_mst_init, angular_error_threshold_degrees=0.1
            )
        )

    def test_initialization_big(self):
        """Test that the result of Shonan is not dependent on the initialization on a bigger dataset."""
        gt_data = io_utils.read_bal(str(LARGE_PROBLEM_BAL_FILE))
        poses = gt_data.get_camera_poses()[:15]
        pairs: List[Tuple[int, int]] = []
        for i in range(len(poses)):
            for j in range(i + 1, min(i + 5, len(poses))):
                pairs.append((i, j))

        i2Ri1_dict_noisefree, _ = sample_poses.convert_data_for_rotation_averaging(
            poses, sample_poses.generate_relative_from_global(poses, pairs)
        )
        v_corr_idxs = {pair: _generate_corr_idxs(random.randint(1, 10)) for pair in i2Ri1_dict_noisefree.keys()}

        # Add noise to the relative rotations
        i2Ri1_dict_noisy = {
            pair: i2Ri1 * rotation_util.random_rotation(angle_scale_factor=0.5)
            for pair, i2Ri1 in i2Ri1_dict_noisefree.items()
        }

        wRi_computed_with_random_init = self.obj.run_rotation_averaging(
            num_images=len(poses),
            i2Ri1_dict=i2Ri1_dict_noisy,
            i1Ti2_priors={},
            v_corr_idxs=v_corr_idxs,
        )

        shonan_mst_init = ShonanRotationAveraging(use_mst_init=True)
        wRi_computed_with_mst_init = shonan_mst_init.run_rotation_averaging(
            num_images=len(poses),
            i2Ri1_dict=i2Ri1_dict_noisy,
            i1Ti2_priors={},
            v_corr_idxs=v_corr_idxs,
        )

        self.assertTrue(
            geometry_comparisons.compare_rotations(
                wRi_computed_with_random_init, wRi_computed_with_mst_init, angular_error_threshold_degrees=0.1
            )
        )


def _generate_corr_idxs(num_corrs: int) -> np.ndarray:
    return np.random.randint(low=0, high=10000, size=(num_corrs, 2))


if __name__ == "__main__":
    unittest.main()
