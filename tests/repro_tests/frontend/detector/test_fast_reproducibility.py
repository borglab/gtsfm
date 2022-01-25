""""Reproducibility tests for FAST detector.

Authors: Ayush Baid
"""
import tests.repro_tests.frontend.detector.test_detector_reproducibility_base as test_detector_reproducibility_base
from gtsfm.frontend.detector.fast import Fast


class TestFastReproducibility(test_detector_reproducibility_base.DetectorReproducibilityTestBase):
    def setUp(self) -> None:
        super().setUp(detector=Fast())
