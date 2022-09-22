"""Tests for SIFT detector descriptor

Authors: Ayush Baid
"""
import unittest

import tests.frontend.detector_descriptor.test_detector_descriptor_base as test_detector_descriptor_base
from gtsfm.frontend.detector.detector_from_joint_detector_descriptor import DetectorFromDetectorDescriptor
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor


class TestSIFTDetectorDescriptor(test_detector_descriptor_base.TestDetectorDescriptorBase):
    """Main test class for detector-description combination base class in frontend.

    All unit test functions defined in TestDetectorDescriptorBase are run automatically.
    """

    def setUp(self):
        """Setup the attributes for the tests."""
        super().setUp()
        self.detector_descriptor = SIFTDetectorDescriptor()

        # explicitly set the detector
        self.detector = DetectorFromDetectorDescriptor(self.detector_descriptor)


if __name__ == "__main__":
    unittest.main()
