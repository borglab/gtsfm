"""
Unit test for RootSIFT descriptor.

Authors: Ayush Baid
"""
import tests.frontend.descriptor.test_descriptor_base as test_descriptor_base
from frontend.descriptor.rootsift import RootSIFT


class TestRootSIFT(test_descriptor_base.TestDescriptorBase):
    """
    Unit tests for RootSIFT descriptor."""

    def setUp(self):
        super().setUp()

        self.descriptor = RootSIFT()
