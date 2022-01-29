""""Reproducibility tests for DoG detector.

Authors: Ayush Baid
"""
import tests.repro_tests.frontend.detector.test_detector_reproducibility_base as test_detector_reproducibility_base
from gtsfm.frontend.detector.dog import DoG


class TestDoGReproducibility(test_detector_reproducibility_base.DetectorReproducibilityTestBase):
    def setUp(self) -> None:
        super().setUp(detector=DoG())
