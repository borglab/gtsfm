"""Reproducibility tests for 1DSFM translation averaging, using input data from sift-front-end on skydio-32 dataset.

Authors: Ayush Baid
"""
import pickle
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gtsam
from gtsam import BinaryMeasurementsUnit3, Point3, Pose3, Rot3, Unit3

import gtsfm.averaging.translation.averaging_1dsfm as averaging_1dsfm
import gtsfm.utils.geometry_comparisons as geometry_comparisons
from gtsfm.averaging.translation.averaging_1dsfm import TranslationAveraging1DSFM
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from tests.repro_tests.test_repro_base import ReproducibilityTestBase

NUM_IMAGES_IN_INPUT = 32
# Paths for global rotations and relative pairwise unit translations from SIFT front-end on Skydio-32
TEST_DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"
RELATIVE_UNIT_TRANSLATIONS_PATH = (
    TEST_DATA_ROOT_PATH / "reproducibility" / "inputs" / "relative_unit_translations_skydio32.pkl"
)
GLOBAL_ROTATIONS_PATH = TEST_DATA_ROOT_PATH / "reproducibility" / "inputs" / "global_rotations_post_shonan_skydio32.pkl"


class Test1DSFMReproducibility(ReproducibilityTestBase, unittest.TestCase):
    """Reproducibility test for the final optimization result of 1DSFM."""

    def setUp(self) -> None:
        super().setUp()
        self._1dsfm_obj: TranslationAveragingBase = TranslationAveraging1DSFM()
        with open(str(RELATIVE_UNIT_TRANSLATIONS_PATH), "rb") as f:
            self._relative_unit_translations_input: Dict[Tuple[int, int], Unit3] = pickle.load(f)
        with open(str(GLOBAL_ROTATIONS_PATH), "rb") as f:
            self._global_rotations_input: List[Optional[Rot3]] = pickle.load(f)

    def run_once(self) -> List[Optional[Point3]]:
        return self._1dsfm_obj.run(
            num_images=NUM_IMAGES_IN_INPUT,
            i2Ui1_dict=self._relative_unit_translations_input,
            wRi_list=self._global_rotations_input,
        )[0]

    def assert_results(self, results_a: List[Optional[Point3]], results_b: List[Optional[Point3]]) -> None:
        poses_a = [Pose3(R, t) if t is not None else None for R, t in zip(self._global_rotations_input, results_a)]
        poses_b = [Pose3(R, t) if t is not None else None for R, t in zip(self._global_rotations_input, results_b)]
        self.assertTrue(geometry_comparisons.compare_global_poses(poses_a, poses_b))


class Test1DSFMInlierMaskReproducibility(ReproducibilityTestBase, unittest.TestCase):
    """Reproducibility test for the inlier mask (on input relative directions) computed by 1D-SFM."""

    def setUp(self) -> None:
        super().setUp()
        self._1dsfm_obj: TranslationAveragingBase = TranslationAveraging1DSFM()
        with open(str(RELATIVE_UNIT_TRANSLATIONS_PATH), "rb") as f:
            relative_unit_translations_input: Dict[Tuple[int, int], Unit3] = pickle.load(f)
        with open(str(GLOBAL_ROTATIONS_PATH), "rb") as f:
            global_rotations_input: List[Optional[Rot3]] = pickle.load(f)
        noise_model = gtsam.noiseModel.Isotropic.Sigma(
            averaging_1dsfm.NOISE_MODEL_DIMENSION, averaging_1dsfm.NOISE_MODEL_SIGMA
        )
        huber_loss = gtsam.noiseModel.mEstimator.Huber.Create(averaging_1dsfm.HUBER_LOSS_K)
        noise_model = gtsam.noiseModel.Robust.Create(huber_loss, noise_model)

        self._w_i2Ui1_measurements: BinaryMeasurementsUnit3 = (
            averaging_1dsfm.cast_to_measurements_variable_in_global_coordinate_frame(
                relative_unit_translations_input, global_rotations_input, noise_model
            )
        )

    def run_once(self) -> Set[Tuple[int, int]]:
        return self._1dsfm_obj.compute_inlier_mask(self._w_i2Ui1_measurements)

    def assert_results(self, results_a: Set[Tuple[int, int]], results_b: Set[Tuple[int, int]]) -> None:
        self.assertEqual(results_a, results_b)
