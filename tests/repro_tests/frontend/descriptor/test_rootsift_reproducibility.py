""""Reproducibility tests for SIFT descriptor.

Authors: Ayush Baid
"""
import tests.repro_tests.frontend.descriptor.test_descriptor_reproducibility_base as base_test
from gtsfm.frontend.descriptor.rootsift import RootSIFTDescriptor


class TestRootSIFTReproducibility(base_test.DescriptorReproducibilityTestBase):
    def setUp(self) -> None:
        super().setUp(descriptor=RootSIFTDescriptor())
