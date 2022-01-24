""""Reproducibility tests for SIFT descriptor.

Authors: Ayush Baid
"""
import tests.repro_tests.frontend.descriptor.test_descriptor_reproducibility_base as test_descriptor_reproducibility_base
from gtsfm.frontend.descriptor.rootsift import RootSIFTDescriptor


class TestRootSIFTReproducibility(test_descriptor_reproducibility_base.DescriptorReproducibilityTestBase):
    def setUp(self) -> None:
        super().setUp(descriptor=RootSIFTDescriptor())
