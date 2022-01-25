"""Reproducibility tests for 1DSFM translation averaging.

Authors: Ayush Baid
"""
import pickle
import unittest
from pathlib import Path
from typing import List, Optional, Set, Tuple

import gtsam
from gtsam import BinaryMeasurementUnit3, BinaryMeasurementsUnit3, Point3, Pose3, Unit3

import gtsfm.utils.geometry_comparisons as geometry_comparisons
from gtsfm.averaging.translation.averaging_1dsfm import (
    TranslationAveraging1DSFM,
    HUBER_LOSS_K,
    NOISE_MODEL_DIMENSION,
    NOISE_MODEL_SIGMA,
)
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from tests.repro_tests.test_repro_base import ReproducibilityTestBase

TEST_DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"
RELATIVE_UNIT_TRANSLATIONS_PATH = (
    TEST_DATA_ROOT_PATH / "reproducibility" / "inputs" / "relative_unit_translations_skydio32.pkl"
)
GLOBAL_ROTATIONS_PATH = TEST_DATA_ROOT_PATH / "reproducibility" / "inputs" / "global_rotations_post_shonan_skydio32.pkl"

ROT3_DIFF_ANGLE_THRESHOLD_DEG = 2


class Test1DSFMReproducibility(ReproducibilityTestBase, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        with open(str(RELATIVE_UNIT_TRANSLATIONS_PATH), "rb") as f:
            self._relative_unit_translations_input = pickle.load(f)
        with open(str(GLOBAL_ROTATIONS_PATH), "rb") as f:
            self._global_rotations_input = pickle.load(f)
        self._1dsfm_obj: TranslationAveragingBase = TranslationAveraging1DSFM()

    def run_once(self) -> List[Optional[Point3]]:
        return self._1dsfm_obj.run(
            num_images=32, i2Ui1_dict=self._relative_unit_translations_input, wRi_list=self._global_rotations_input
        )[0]

    def assert_results(self, results_a: List[Optional[Point3]], results_b: List[Optional[Point3]]) -> None:
        poses_a = [Pose3(r, t) if t is not None else None for r, t in zip(self._global_rotations_input, results_a)]
        poses_b = [Pose3(r, t) if t is not None else None for r, t in zip(self._global_rotations_input, results_b)]
        self.assertTrue(geometry_comparisons.compare_global_poses(poses_a, poses_b))


class Test1DSFMInlierMaskReproducibility(ReproducibilityTestBase, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._1dsfm_obj: TranslationAveragingBase = TranslationAveraging1DSFM()
        with open(str(RELATIVE_UNIT_TRANSLATIONS_PATH), "rb") as f:
            relative_unit_translations_input = pickle.load(f)
        with open(str(GLOBAL_ROTATIONS_PATH), "rb") as f:
            global_rotations_input = pickle.load(f)
        noise_model = gtsam.noiseModel.Isotropic.Sigma(NOISE_MODEL_DIMENSION, NOISE_MODEL_SIGMA)
        huber_loss = gtsam.noiseModel.mEstimator.Huber.Create(HUBER_LOSS_K)
        noise_model = gtsam.noiseModel.Robust.Create(huber_loss, noise_model)

        self._w_i2Ui1_measurements = BinaryMeasurementsUnit3()
        for (i1, i2), i2Ui1 in relative_unit_translations_input.items():
            if i2Ui1 is not None and global_rotations_input[i2] is not None:
                self._w_i2Ui1_measurements.append(
                    BinaryMeasurementUnit3(
                        i2, i1, Unit3(global_rotations_input[i2].rotate(i2Ui1.point3())), noise_model
                    )
                )

    def run_once(self) -> Set[Tuple[int, int]]:
        return self._1dsfm_obj.get_inlier_mask_for_direction_measurements(self._w_i2Ui1_measurements)

    def assert_results(self, results_a: Set[Tuple[int, int]], results_b: Set[Tuple[int, int]]) -> None:
        self.assertEqual(results_a, results_b)
