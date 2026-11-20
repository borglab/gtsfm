"""Reproducibility tests for the combined OpenCV RootSIFT extractor."""

import tests.repro_tests.frontend.detector_descriptor.test_detector_descriptor_reproducibility_base as base_test
from gtsfm.frontend.detector_descriptor.rootsift import RootSIFTDetectorDescriptor


class TestRootSIFTReproducibility(base_test.DetectorDescriptorReproducibilityTestBase):
    """Verify repeatability of both keypoints and normalized descriptors."""

    def setUp(self) -> None:
        """Use the combined RootSIFT extractor for repeated extraction."""
        super().setUp(detector_descriptor=RootSIFTDetectorDescriptor())
