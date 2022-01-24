"""Reproducibility tests for 1DSFM translation averaging.

Authors: Ayush Baid
"""
import pickle
import unittest
from pathlib import Path
from typing import List, Optional

from gtsam import Point3, Pose3

import gtsfm.utils.geometry_comparisons as geometry_comparisons
from gtsfm.averaging.translation.averaging_1dsfm import TranslationAveraging1DSFM
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
        poses_a = [Pose3(r, t) for r, t in zip(self._global_rotations_input, results_a) if t is not None]
        poses_b = [Pose3(r, t) for r, t in zip(self._global_rotations_input, results_b) if t is not None]
        self.assertTrue(geometry_comparisons.compare_global_poses(poses_a, poses_b))
