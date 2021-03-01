"""Tests for translation averaging base class.

Authors: Ayush Baid
"""
import pickle
import unittest
from typing import Dict, List, Tuple

import dask
from gtsam import Cal3_S2, Point3, Rot3, Unit3
from gtsam.examples import SFMdata

import gtsfm.utils.geometry_comparisons as geometry_comparisons
import tests.data.sample_poses as sample_poses
from gtsfm.averaging.translation.dummy_translation_averaging import DummyTranslationAveraging

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
        self.assertTrue(
            geometry_comparisons.align_and_compare_translations(
                wti_computed, wti_expected, RELATIVE_ERROR_THRESHOLD, ABSOLUTE_ERROR_THRESHOLD
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

    def test_line_large_edges(self):
        """Tests for 3 poses in a line, with large translations between them."""
        wRi_list, i2Ui1_dict, wti_expected = sample_poses.convert_data_for_translation_averaging(
            sample_poses.LINE_LARGE_EDGES_GLOBAL_POSES, sample_poses.LINE_LARGE_EDGES_RELATIVE_POSES
        )
        self.__execute_test(i2Ui1_dict, wRi_list, wti_expected)

    def test_line_small_edges(self):
        """Tests for 3 poses in a line, with small translations between them."""
        wRi_list, i2Ui1_dict, wti_expected = sample_poses.convert_data_for_translation_averaging(
            sample_poses.LINE_SMALL_EDGES_GLOBAL_POSES, sample_poses.LINE_SMALL_EDGES_RELATIVE_POSES
        )
        self.__execute_test(i2Ui1_dict, wRi_list, wti_expected)

    # def test_panorama(self):
    #     """Tests for 3 poses in a panorama configuration (large rotations at the same location)."""
    #     wRi_list, i2Ui1_dict, wti_expected = sample_poses.convert_data_for_translation_averaging(
    #         sample_poses.PANORAMA_GLOBAL_POSES, sample_poses.PANORAMA_RELATIVE_POSES
    #     )
    #     self.__execute_test(i2Ui1_dict, wRi_list, wti_expected)

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
        expected_wti_list = self.obj.run(len(wRi_list), i2Ui1_dict, wRi_list)

        # form computation graph and execute
        i2Ui1_graph = dask.delayed(i2Ui1_dict)
        wRi_graph = dask.delayed(wRi_list)
        computation_graph = self.obj.create_computation_graph(len(wRi_list), i2Ui1_graph, wRi_graph)
        with dask.config.set(scheduler="single-threaded"):
            wti_list = dask.compute(computation_graph)[0]
        # compare the entries
        self.assertTrue(
            geometry_comparisons.align_and_compare_translations(
                wti_list, expected_wti_list, RELATIVE_ERROR_THRESHOLD, ABSOLUTE_ERROR_THRESHOLD
            )
        )

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")


if __name__ == "__main__":
    unittest.main()
