"""Tests for Shonan rotation averaging.

Authors: Ayush Baid, John Lambert
"""
import pickle
import unittest
from typing import Dict, List, Tuple

import dask
import numpy as np
from gtsam import Pose3, Rot3

import gtsfm.utils.geometry_comparisons as geometry_comparisons
import tests.data.sample_poses as sample_poses
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging
from gtsfm.common.pose_prior import PosePrior, PosePriorType

ROTATION_ANGLE_ERROR_THRESHOLD_DEG = 2


class TestShonanRotationAveraging(unittest.TestCase):
    """Test class for Shonan rotation averaging.

    All unit test functions defined in TestRotationAveragingBase are run automatically.
    """

    def setUp(self):
        super().setUp()

        self.obj: RotationAveragingBase = ShonanRotationAveraging()

    def __execute_test(self, i2Ri1_input: Dict[Tuple[int, int], Rot3], wRi_expected: List[Rot3]) -> None:
        """Helper function to run the averagaing and assert w/ expected.

        Args:
            i2Ri1_input: relative rotations, which are input to the algorithm.
            wRi_expected: expected global rotations.
        """
        i1Ti2_priors: Dict[Tuple[int, int], PosePrior] = {}
        v_corr_idxs = _create_dummy_correspondences(i2Ri1_input)
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

    def test_simple(self):
        """Test a simple case with three relative rotations."""

        i2Ri1_dict = {
            (1, 0): Rot3.RzRyRx(0, np.deg2rad(30), 0),
            (2, 1): Rot3.RzRyRx(0, 0, np.deg2rad(20)),
        }

        expected_wRi_list = [
            Rot3.RzRyRx(0, 0, 0),
            Rot3.RzRyRx(0, np.deg2rad(30), 0),
            i2Ri1_dict[(1, 0)].compose(i2Ri1_dict[(2, 1)]),
        ]

        self.__execute_test(i2Ri1_dict, expected_wRi_list)

    def test_simple_with_prior(self):
        """Test a simple case with 1 measurement and a single pose prior."""
        expected_wRi_list = [Rot3.RzRyRx(0, 0, 0), Rot3.RzRyRx(0, np.deg2rad(30), 0), Rot3.RzRyRx(np.deg2rad(30), 0, 0)]

        i2Ri1_dict = {
            (1, 0): Rot3.RzRyRx(0, np.deg2rad(30), 0),
        }

        expected_0R2 = expected_wRi_list[0].between(expected_wRi_list[2])
        i1Ti2_priors = {
            (0, 2): PosePrior(
                value=Pose3(expected_0R2, np.zeros((3,))),
                covariance=np.eye(6) * 1e-5,
                type=PosePriorType.SOFT_CONSTRAINT,
            )
        }
        v_corr_idxs = _create_dummy_correspondences(i2Ri1_dict)
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

        i2Ri1_graph = dask.delayed(i2Ri1_dict)

        # Use the GTSAM API directly (without dask) for rotation averaging
        i1Ti2_priors: Dict[Tuple[int, int], PosePrior] = {}
        v_corr_idxs = _create_dummy_correspondences(i2Ri1_dict)
        expected_wRi_list = self.obj.run_rotation_averaging(num_poses, i2Ri1_dict, i1Ti2_priors, v_corr_idxs)

        # Use dask's computation graph
        gt_wTi_list = [None] * len(expected_wRi_list)
        rotations_graph, _ = self.obj.create_computation_graph(
            num_poses, i2Ri1_graph, i1Ti2_priors, v_corr_idxs=v_corr_idxs, gt_wTi_list=gt_wTi_list
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

        # assume pose 0 is orphaned in the visibility graph
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

        relative_pose_priors: Dict[Tuple[int, int], PosePrior] = {}
        v_corr_idxs = _create_dummy_correspondences(i2Ri1_input)
        wRi_computed = self.obj.run_rotation_averaging(num_images, i2Ri1_input, relative_pose_priors, v_corr_idxs)
        wRi_expected = [None, wTi1.rotation(), wTi2.rotation(), wTi3.rotation()]
        self.assertTrue(
            geometry_comparisons.compare_rotations(wRi_computed, wRi_expected, angular_error_threshold_degrees=0.1)
        )


def _create_dummy_correspondences(i2Ri1_dict: Dict[Tuple[int, int], Rot3]) -> Dict[Tuple[int, int], np.ndarray]:
    """Create dummy verified correspondences for each edge in view graph."""
    # Assume image has shape (img_h, img_w) = (1000,1000)
    img_h = 1000
    v_corr_idxs_dict = {
        (i1, i2): np.random.randint(low=0, high=img_h, size=(i1 + i2, 2)) for i1, i2 in i2Ri1_dict.keys()
    }
    return v_corr_idxs_dict


if __name__ == "__main__":
    unittest.main()
