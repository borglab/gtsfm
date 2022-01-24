"""Reproducibility tests for Shonan rotation averaging.

Authors: Ayush Baid
"""
import pickle
import unittest
from pathlib import Path
from typing import List, Optional

from gtsam import Rot3

import gtsfm.utils.geometry_comparisons as geometry_comparisons
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging
from tests.repro_tests.test_repro_base import ReproducibilityTestBase

TEST_DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"
RELATIVE_ROTATIONS_PATH = TEST_DATA_ROOT_PATH / "reproducibility" / "inputs" / "relative_rotations_skydio32.pkl"

ROT3_DIFF_ANGLE_THRESHOLD_DEG = 2


class TestShonanAveragingReproducibility(ReproducibilityTestBase, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        with open(str(RELATIVE_ROTATIONS_PATH), "rb") as f:
            self._input = pickle.load(f)
        self._shonan_obj: RotationAveragingBase = ShonanRotationAveraging()

    def run_once(self) -> List[Optional[Rot3]]:
        return self._shonan_obj.run(num_images=32, i2Ri1_dict=self._input)

    def assert_results(self, results_a: List[Optional[Rot3]], results_b: List[Optional[Rot3]]) -> None:
        self.assertTrue(geometry_comparisons.compare_rotations(results_a, results_b, ROT3_DIFF_ANGLE_THRESHOLD_DEG))
