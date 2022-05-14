"""Tests for 1DSFM translation averaging.

Authors: Ayush Baid
"""
import pickle
from statistics import covariance
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import dask
import gtsam
import numpy as np
from gtsam import Cal3_S2, Point3, Pose3, Rot3, Unit3
from gtsam.examples import SFMdata
from gtsfm.common.pose_prior import PosePrior

import gtsfm.utils.geometry_comparisons as geometry_comparisons

import gtsfm.utils.sample_poses as sample_poses
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

        self.obj: TranslationAveraging1DSFM = TranslationAveraging1DSFM()

    def test_augmentation(self):
        """Tests that measurements for betweenTranslations are augmented to input measurements."""
        input_measurements = gtsam.BinaryMeasurementsUnit3()
        INPUT_NOISE_MODEL = gtsam.noiseModel.Isotropic(2, 1e-2)
        input_measurements.append(gtsam.BinaryMeasurementUnit3(0, 1, Unit3(Point3(1, 0, 0), INPUT_NOISE_MODEL)))
        input_measurements.append(gtsam.BinaryMeasurementUnit3(0, 2, Unit3(Point3(0, 1, 0), INPUT_NOISE_MODEL)))
        input_measurements.append(gtsam.BinaryMeasurementUnit3(1, 2, Unit3(Point3(0, 0, 1), INPUT_NOISE_MODEL)))

        priors = gtsam.BinaryMeasurementsPoint3()
        PRIOR_NOISE_MODEL = gtsam.noiseModel.Isotropic(3, 1e-2)
        priors.append(gtsam.BinaryMeasurementPoint3(0, 2, Point3(1, 1, 0), PRIOR_NOISE_MODEL))
        priors.append(gtsam.BinaryMeasurementPoint3(2, 3, Point3(1, 0, 1), PRIOR_NOISE_MODEL))

        augmented_measurements = self.obj.augment(input_measurements, priors, INPUT_NOISE_MODEL)
        self.assertEqual(len(augmented_measurements), 4)
        augmented_measurements_dict = {}
        for idx in range(len(augmented_measurements)):
            measurement = augmented_measurements[idx]
            augmented_measurements_dict[(measurement.key1(), measurement.key2())] = measurement.measured()
        self.assertSetEqual(set(augmented_measurements_dict.keys()), set([(0, 1), (0, 2), (1, 2), (2, 3)]))
        self.assertEqual(augmented_measurements_dict[2, 3], Unit3(Point3(1, 0, 1)))

    def test_convert_prior_to_world_frame(self):
        """Test the helper method for transforming betweenTranslations to world frame."""
        # Rotate i1 by 90 degrees in Z, and translate by (1, 1, 1)
        wRi1 = Rot3.Rz(90)
        i1Ti2 = Pose3(Rot3(), Point3(2, 0, 0))  # Identity rotation, X axis.

        covariance = np.zeros((6, 6))
        for i in range(6):
            covariance[i, i] = 1.0 if i is not 3 else 2.0
        i1Ti2_prior = PosePrior(value=i1Ti2, covariance=covariance)

        wRi_list = [Rot3(), wRi1, Rot3()]   # identity for 0 and 2, wRi1 for 1
        w_i1ti2_priors = self.obj._get_prior_measurements_in_world_frame({(1, 2): i1Ti2_prior}, wRi_list)
        # Check length
        self.assertEqual(len(w_i1ti2_priors), 1)

        # Check keys
        actual_i1 = w_i1ti2_priors[0].key1()
        actual_i2 = w_i1ti2_priors[0].key2()
        self.assertEqual(actual_i1, 1)
        self.assertEqual(actual_i2, 2)

        # Check value
        actual_w_i1ti2 = w_i1ti2_priors[0].measured()
        self.assertEqual(actual_w_i1ti2, Point3(0, 2, 0))

        # Check covariance
        actual_covariance = w_i1ti2_priors[0].noise_model().covariance()
        expected_covariance = np.eye(3)
        expected_covariance[1, 1] = 2.0
        self.assertEqual(actual_covariance, expected_covariance)
        

    def __execute_test(
        self, i2Ui1_input: Dict[Tuple[int, int], Unit3], wRi_input: List[Rot3], wti_expected: List[Point3]
    ) -> None:
        """Helper function to run the averagaing and assert w/ expected."""

        wti_computed, _ = self.obj.run_translation_averaging(len(wRi_input), i2Ui1_input, wRi_input)

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
        wti_expected, _ = self.obj.run_translation_averaging(len(wRi_list), i2Ui1_dict, wRi_list)

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
        wti_computed, _ = self.obj.run_translation_averaging(len(wRi_input), i2Ui1_input, wRi_input)

        assert len(wti_computed) == 5
        assert wti_computed[-1] is None


if __name__ == "__main__":
    unittest.main()
