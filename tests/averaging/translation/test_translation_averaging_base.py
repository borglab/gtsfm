"""Tests for translation averaging base class.

Authors: Ayush Baid
"""
import pickle
import unittest
from typing import Dict, List, Optional, Tuple

import dask
import numpy as np
from gtsam import Cal3_S2, Point3, Pose3, Rot3, Unit3
from gtsam.examples import SFMdata

import gtsfm.utils.geometry_comparisons as geometry_comparisons
import tests.data.sample_poses as sample_poses
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase

RELATIVE_ERROR_THRESHOLD = 3e-2
ABSOLUTE_ERROR_THRESHOLD = 3e-1


class TestTranslationAveragingBase(unittest.TestCase):
    """Main tests for translation averaging base class.

    This class should be inherited by all unit tests for translation averaging implementations.
    """

    def setUp(self):
        super().setUp()

        self.obj = DummyTranslationAveraging()

    def __execute_test(
        self, i2Ui1_input: Dict[Tuple[int, int], Unit3], wRi_input: List[Rot3], wti_expected: List[Point3]
    ) -> None:
        """Helper function to run the averagaing and assert w/ expected.

        Args:
            i2Ri1_input (Dict[Tuple[int, int], Rot3]): [description]
            wRi_expected (List[Rot3]): [description]
        """
        if isinstance(self.obj, DummyTranslationAveraging):
            self.skipTest("Test case invalid for dummy class")

        wti_computed = self.obj.run(len(wRi_input), i2Ui1_input, wRi_input)

        wTi_computed = [Pose3(wRi, wti) for wRi, wti in zip(wRi_input, wti_computed)]
        wTi_expected = [Pose3(wRi, wti) for wRi, wti in zip(wRi_input, wti_expected)]
        self.assertTrue(
            geometry_comparisons.compare_global_poses(
                wTi_computed, wTi_expected, RELATIVE_ERROR_THRESHOLD, ABSOLUTE_ERROR_THRESHOLD
            )
        )

    def test_circle_two_edges(self):
        """Tests for 4 poses in a circle, with a pose connected to its immediate neighborhood."""
        wRi_list, i2Ui1_dict, wti_expected = sample_poses.convert_data_for_translation_averaging(
            sample_poses.CIRCLE_TWO_EDGES_GLOBAL_POSES, sample_poses.CIRCLE_TWO_EDGES_RELATIVE_POSES
        )
        self.__execute_test(i2Ui1_dict, wRi_list, wti_expected)

    def test_circle_all_edges(self):
        """Tests for 4 poses in a circle, with a pose connected all others."""
        wRi_list, i2Ui1_dict, wti_expected = sample_poses.convert_data_for_translation_averaging(
            sample_poses.CIRCLE_ALL_EDGES_GLOBAL_POSES, sample_poses.CIRCLE_ALL_EDGES_RELATIVE_POSES
        )
        self.__execute_test(i2Ui1_dict, wRi_list, wti_expected)

    # deprecating as underconstrained problem
    # def test_line_large_edges(self):
    #     """Tests for 3 poses in a line, with large translations between them."""
    #     wRi_list, i2Ui1_dict, wti_expected = sample_poses.convert_data_for_translation_averaging(
    #         sample_poses.LINE_LARGE_EDGES_GLOBAL_POSES, sample_poses.LINE_LARGE_EDGES_RELATIVE_POSES
    #     )
    #     self.__execute_test(i2Ui1_dict, wRi_list, wti_expected)

    def test_line_small_edges(self):
        """Tests for 3 poses in a line, with small translations between them."""
        wRi_list, i2Ui1_dict, wti_expected = sample_poses.convert_data_for_translation_averaging(
            sample_poses.LINE_SMALL_EDGES_GLOBAL_POSES, sample_poses.LINE_SMALL_EDGES_RELATIVE_POSES
        )
        self.__execute_test(i2Ui1_dict, wRi_list, wti_expected)

    def test_panorama(self):
        """Tests for 3 poses in a panorama configuration (large rotations at the same location)."""
        wRi_list, i2Ui1_dict, wti_expected = sample_poses.convert_data_for_translation_averaging(
            sample_poses.PANORAMA_GLOBAL_POSES, sample_poses.PANORAMA_RELATIVE_POSES
        )
        self.__execute_test(i2Ui1_dict, wRi_list, wti_expected)

    def test_computation_graph(self):
        """Test the dask computation graph execution using a valid collection of relative unit-translations."""

        """Test a simple case with 8 camera poses.

        The camera poses are arranged on the circle and point towards the center
        of the circle. The poses of 8 cameras are obtained from SFMdata and the
        unit translations directions between some camera pairs are computed from their global translations.

        This test is copied from GTSAM's TranslationAveragingExample.
        """

        fx, fy, s, u0, v0 = 50.0, 50.0, 0.0, 50.0, 50.0
        expected_wTi_list = SFMdata.createPoses(Cal3_S2(fx, fy, s, u0, v0))

        wRi_list = [x.rotation() for x in expected_wTi_list]

        # create relative translation directions between a pose index and the
        # next two poses
        i2Ui1_dict = {}
        for i1 in range(len(expected_wTi_list) - 1):
            for i2 in range(i1 + 1, min(len(expected_wTi_list), i1 + 3)):
                # create relative translations using global R and T.
                i2Ui1_dict[(i1, i2)] = Unit3(expected_wTi_list[i2].between(expected_wTi_list[i1]).translation())

        # use the `run` API to get expected results
        wti_expected = self.obj.run(len(wRi_list), i2Ui1_dict, wRi_list)

        # form computation graph and execute
        i2Ui1_graph = dask.delayed(i2Ui1_dict)
        wRi_graph = dask.delayed(wRi_list)
        computation_graph = self.obj.create_computation_graph(len(wRi_list), i2Ui1_graph, wRi_graph)
        with dask.config.set(scheduler="single-threaded"):
            wti_computed = dask.compute(computation_graph)[0]
        wTi_computed = [Pose3(wRi, wti) for wRi, wti in zip(wRi_list, wti_computed)]
        wTi_expected = [Pose3(wRi, wti) for wRi, wti in zip(wRi_list, wti_expected)]
        self.assertTrue(
            geometry_comparisons.compare_global_poses(
                wTi_computed, wTi_expected, RELATIVE_ERROR_THRESHOLD, ABSOLUTE_ERROR_THRESHOLD
            )
        )

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")


class DummyTranslationAveraging(TranslationAveragingBase):
    """Assigns random unit-translations to each pose."""

    def run(
        self,  # pylint: disable=unused-argument
        num_images: int,
        i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],  # pylint: disable=unused-argument
        wRi_list: List[Optional[Rot3]],
        scale_factor: float = 1.0,
    ) -> List[Optional[Point3]]:
        """Run the translation averaging.

        Args:
            num_images: number of camera poses.
            i2Ui1_dict: relative unit-trans as dictionary (i1, i2): i2Ui1.
            wRi_list: global rotations for each camera pose in the world coordinates.
            scale_factor: non-negative global scaling factor.

        Returns:
            Global translation wti for each camera pose. The number of entries in the list is `num_images`. The list
                may contain `None` where the global translations could not be computed (either underconstrained system
                or ill-constrained system).
        """

        if len(wRi_list) == 0:
            return []

        # create the random seed using relative rotations
        seed_rotation = wRi_list[0]
        np.random.seed(int(1000 * seed_rotation.xyz()[0]) % (2 ^ 32))

        # generate dummy output
        results = [None] * num_images
        for idx in range(num_images):
            if wRi_list[idx] is not None:
                random_vector = np.random.rand(3)
                results[idx] = scale_factor * Unit3(random_vector).point3()

        return results


if __name__ == "__main__":
    unittest.main()
