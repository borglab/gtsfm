""""Reproducibility tests for SIFT descriptor.

Authors: Ayush Baid
"""
import tests.repro_tests.frontend.detector_descriptor.test_detector_descriptor_reproducibility_base as base_test
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor


class TestSIFTDetectorDescriptorReproducibility(base_test.DetectorDescriptorReproducibilityTestBase):
    def setUp(self) -> None:
        super().setUp(detector_descriptor=SIFTDetectorDescriptor())
