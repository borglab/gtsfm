""""Reproducibility tests for descriptors.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path

import numpy as np

import gtsfm.utils.io as io_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.descriptor.descriptor_base import DescriptorBase
from gtsfm.frontend.descriptor.sift import SIFTDescriptor
from gtsfm.frontend.detector.detector_base import DetectorBase
from gtsfm.frontend.detector.dog import DoG
from tests.repro_tests.test_repro_base import ReproducibilityTestBase

# defining the path for test data
TEST_DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"
IMG_PATH = TEST_DATA_ROOT_PATH / "set1_lund_door" / "images" / "DSC_0001.JPG"


class DescriptorReproducibilityTestBase(ReproducibilityTestBase, unittest.TestCase):
    def setUp(self, descriptor: DescriptorBase = SIFTDescriptor()) -> None:
        super().setUp()
        detector: DetectorBase = DoG()
        self._image: Image = io_utils.load_image(str(IMG_PATH))
        self._keypoints: Keypoints = detector.detect(self._image)
        self._descriptor: DescriptorBase = descriptor

    def run_once(self) -> np.ndarray:
        return self._descriptor.describe(self._image, self._keypoints)

    def assert_results(self, results_a: np.ndarray, results_b: np.ndarray) -> None:
        np.testing.assert_allclose(results_a, results_b)
