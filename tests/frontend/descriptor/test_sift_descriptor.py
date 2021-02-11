"""Unit test for SIFT descriptor.

Authors: Ayush Baid
"""
import unittest

import tests.frontend.descriptor.test_descriptor_base as test_descriptor_base
from gtsfm.frontend.descriptor.sift import SIFTDescriptor


class TestSIFTDescriptor(test_descriptor_base.TestDescriptorBase):
    """Unit tests for RootSIFT descriptor.

    All unit test functions defined in TestDescriptorBase are run automatically.
    """

    def setUp(self):
        super().setUp()

        self.descriptor = SIFTDescriptor()


if __name__ == "__main__":
    unittest.main()
