""""Reproducibility tests for detector-descriptors.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path
from typing import Tuple

import numpy as np

import gtsfm.utils.io as io_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from tests.repro_tests.test_repro_base import ReproducibilityTestBase

# path for test data, which is one image
TEST_DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"
IMG_PATH = TEST_DATA_ROOT_PATH / "set1_lund_door" / "images" / "DSC_0001.JPG"

DET_DESC_RESULT_TYPE = Tuple[Keypoints, np.ndarray]


class DetectorDescriptorReproducibilityTestBase(ReproducibilityTestBase, unittest.TestCase):
    def setUp(self, detector_descriptor: DetectorDescriptorBase = SIFTDetectorDescriptor()) -> None:
        super().setUp()
        self._input: Image = io_utils.load_image(str(IMG_PATH))
        self._detector_descriptor: DetectorDescriptorBase = detector_descriptor

    def run_once(self) -> DET_DESC_RESULT_TYPE:
        return self._detector_descriptor.detect_and_describe(self._input)

    def assert_results(self, results_a: DET_DESC_RESULT_TYPE, results_b: DET_DESC_RESULT_TYPE) -> None:
        self.assertEqual(results_a[0], results_b[0])
        np.testing.assert_allclose(results_a[1], results_b[1])
