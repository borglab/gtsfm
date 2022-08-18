"""Tests for 1DSFM translation averaging.

Authors: Ayush Baid
"""
import pickle
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import dask
import numpy as np
from gtsam import Cal3_S2, Point3, Pose3, Rot3, Unit3
from gtsam.examples import SFMdata

import gtsfm.utils.geometry_comparisons as geometry_comparisons
import tests.data.sample_poses as sample_poses
from gtsfm.averaging.translation.averaging_1dsfm import TranslationAveraging1DSFM
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.loader.olsson_loader import OlssonLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent / "data"

RELATIVE_ERROR_THRESHOLD = 3e-2
ABSOLUTE_ERROR_THRESHOLD = 3e-1


class TestTranslationAveraging1DSFM(unittest.TestCase):
    """Test class for 1DSFM rotation averaging."""

    def setUp(self):
        super().setUp()

        self.obj: TranslationAveragingBase = TranslationAveraging1DSFM()

    def __execute_test(
        self, i2Ui1_input: Dict[Tuple[int, int], Unit3], wRi_input: List[Rot3], wti_expected: List[Point3]
    ) -> None:
        """Helper function to run the averagaing and assert w/ expected."""

        wti_computed, _ = self.obj.run(len(wRi_input), i2Ui1_input, wRi_input)

        wTi_computed = [Pose3(wRi, wti) for wRi, wti in zip(wRi_input, wti_computed)]
        wTi_expected = [Pose3(wRi, wti) for wRi, wti in zip(wRi_input, wti_expected)]
        self.assertTrue(
            geometry_comparisons.compare_global_poses(
                wTi_computed, wTi_expected, RELATIVE_ERROR_THRESHOLD, ABSOLUTE_ERROR_THRESHOLD
            )
        )

    # deprecating as underconstrained problem
    # def test_circle_two_edges(self):
    #     """Tests for 4 poses in a circle, with a pose connected to its immediate neighborhood."""
    #     wRi_list, i2Ui1_dict, wti_expected = sample_poses.convert_data_for_translation_averaging(
    #         sample_poses.CIRCLE_TWO_EDGES_GLOBAL_POSES, sample_poses.CIRCLE_TWO_EDGES_RELATIVE_POSES
    #     )
    #     self.__execute_test(i2Ui1_dict, wRi_list, wti_expected)

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

    # deprecating as underconstrained problem
    # def test_line_small_edges(self):
    #     """Tests for 3 poses in a line, with small translations between them."""
    #     wRi_list, i2Ui1_dict, wti_expected = sample_poses.convert_data_for_translation_averaging(
    #         sample_poses.LINE_SMALL_EDGES_GLOBAL_POSES, sample_poses.LINE_SMALL_EDGES_RELATIVE_POSES
    #     )
    #     self.__execute_test(i2Ui1_dict, wRi_list, wti_expected)

    def test_panorama(self):
        """Tests for 3 poses in a panorama configuration (large rotations at the same location)."""
        wRi_list, i2Ui1_dict, wti_expected = sample_poses.convert_data_for_translation_averaging(
            sample_poses.PANORAMA_GLOBAL_POSES, sample_poses.PANORAMA_RELATIVE_POSES
        )
        self.__execute_test(i2Ui1_dict, wRi_list, wti_expected)

    def test_lund_door(self):
        """Unit Test on the door dataset."""
        loader = OlssonLoader(str(DATA_ROOT_PATH / "set1_lund_door"), image_extension="JPG")

        # we will use ground truth poses to generate relative rotations and relative unit translations
        wTi_expected_list = [loader.get_camera_pose(x) for x in range(len(loader))]
        wRi_list = [x.rotation() for x in wTi_expected_list]
        wti_expected_list = [x.translation() for x in wTi_expected_list]

        i2Ui1_dict = dict()
        for (i1, i2) in loader.get_valid_pairs():
            i2Ti1 = wTi_expected_list[i2].between(wTi_expected_list[i1])

            i2Ui1_dict[(i1, i2)] = Unit3((i2Ti1.translation()))

        self.__execute_test(i2Ui1_dict, wRi_list, wti_expected_list)

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

        # use the `run` API to get expected results, ignore the metrics
        wti_expected, _ = self.obj.run(len(wRi_list), i2Ui1_dict, wRi_list)

        # form computation graph and execute
        i2Ui1_graph = dask.delayed(i2Ui1_dict)
        wRi_graph = dask.delayed(wRi_list)
        computation_graph = self.obj.create_computation_graph(len(wRi_list), i2Ui1_graph, wRi_graph)
        with dask.config.set(scheduler="single-threaded"):
            wti_computed, _ = dask.compute(computation_graph)[0]
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


class Test1dsfmAllOutliers(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.obj: TranslationAveragingBase = TranslationAveraging1DSFM()

    def test_outlier_case_missing_value(self) -> None:
        """Ensure that a missing `Value` in the 1dsfm result is represented by `None` in the returned entries.
        
        The scenario below will lead to an outlier configuration -- all edges to the node 4 will be rejected
        as outliers, so that Value cannot be cast to Point3 -- it is returned as None.
        
        This test ensures that 1dsfm checks if each Value exists in 1dsfm result, before casting it to a Point3.
        """
        # fmt: off
        wRi_list = [
            np.array(
                [
                    [-0.382164, 0.89195, 0.241612],
                    [-0.505682, 0.0169854, -0.862553],
                    [-0.773458, -0.451815, 0.444551]
                ]),
            np.array(
                [
                    [-0.453335, 0.886803, -0.0898219],
                    [-0.27425, -0.234656, -0.93259],
                    [-0.8481, -0.398142, 0.349584]
                ]),
            np.array(
                [
                    [-0.385656, 0.90387, -0.18517],
                    [0.125519, -0.147431, -0.981076],
                    [-0.914065, -0.4016, -0.0565954]
                ]),
            np.array(
                [
                    [-0.359387, 0.898029, -0.253744],
                    [0.253506, -0.167734, -0.95268],
                    [-0.898096, -0.406706, -0.167375]
                ]),
            np.array(
                [
                    [-0.342447, 0.898333, -0.275186],
                    [0.0881727, -0.260874, -0.961338],
                    [-0.935391, -0.353471, 0.0101272]
                ]),
        ]
        # fmt: on
        wRi_input = [Rot3(wRi) for wRi in wRi_list]

        i2Ui1_input = {
            (0, 1): np.array([0.967948, -0.0290259, 0.24947]),
            (0, 2): np.array([0.906879, -0.000610539, 0.42139]),
            (0, 3): np.array([0.937168, -0.0161865, 0.348502]),
            (0, 4): np.array([-0.975139, 0.0133109, -0.221193]),
            (1, 2): np.array([0.990186, 0.0188153, 0.138484]),
            (1, 3): np.array([0.986072, -0.00746304, 0.166149]),
            (1, 4): np.array([-0.996558, 0.00911097, -0.0823996]),
            (2, 3): np.array([0.990546, -0.0294894, 0.133976]),
            (2, 4): np.array([0.998932, -0.0300599, -0.035099]),
            (3, 4): np.array([0.994791, -0.033332, -0.0963361]),
        }
        i2Ui1_input = {(i, j): Unit3(t) for (i, j), t in i2Ui1_input.items()}
        wti_computed, _ = self.obj.run(len(wRi_input), i2Ui1_input, wRi_input)

        assert len(wti_computed) == 5
        assert wti_computed[-1] is None


if __name__ == "__main__":
    unittest.main()
