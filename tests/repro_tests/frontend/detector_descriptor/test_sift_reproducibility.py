""""Reproducibility tests for SIFT descriptor.

Authors: Ayush Baid
"""
import tests.repro_tests.frontend.detector_descriptor.test_detector_descriptor_reproducibility_base as test_detector_descriptor_reproducibility_base
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor


class TestSIFTDetectorDescriptorReproducibility(
    test_detector_descriptor_reproducibility_base.DetectorDescriptorReproducibilityTestBase
):
    def setUp(self) -> None:
        super().setUp(detector_descriptor=SIFTDetectorDescriptor())
