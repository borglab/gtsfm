"""Tests for the BRISK detector-descriptor.

Authors: John Lambert
"""
import unittest

import tests.frontend.detector_descriptor.test_detector_descriptor_base as test_detector_descriptor_base
from gtsfm.frontend.detector.detector_from_joint_detector_descriptor import DetectorFromDetectorDescriptor
from gtsfm.frontend.detector_descriptor.brisk import BRISKDetectorDescriptor


class TestBRISKDetectorDescriptor(test_detector_descriptor_base.TestDetectorDescriptorBase):
    """Main test class for detector-description combination base class in frontend.

    All unit test functions defined in TestDetectorDescriptorBase are run automatically.
    """

    def setUp(self) -> None:
        """Setup the attributes for the tests."""
        super().setUp()
        self.detector_descriptor = BRISKDetectorDescriptor()

        # explicitly set the detector
        self.detector = DetectorFromDetectorDescriptor(self.detector_descriptor)


if __name__ == "__main__":
    unittest.main()
