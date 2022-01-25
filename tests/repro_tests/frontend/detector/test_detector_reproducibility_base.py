""""Reproducibility tests for detectors.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path

import gtsfm.utils.io as io_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector.detector_base import DetectorBase
from gtsfm.frontend.detector.dog import DoG
from tests.repro_tests.test_repro_base import ReproducibilityTestBase

# defining the path for test data
TEST_DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"
IMG_PATH = TEST_DATA_ROOT_PATH / "set1_lund_door" / "images" / "DSC_0001.JPG"


class DetectorReproducibilityTestBase(ReproducibilityTestBase, unittest.TestCase):
    def setUp(self, detector: DetectorBase = DoG()) -> None:
        super().setUp()
        self._input: Image = io_utils.load_image(str(IMG_PATH))
        self._detector: DetectorBase = detector

    def run_once(self) -> Keypoints:
        return self._detector.detect(self._input)

    def assert_results(self, results_a: Keypoints, results_b: Keypoints) -> None:
        self.assertEqual(results_a, results_b)
