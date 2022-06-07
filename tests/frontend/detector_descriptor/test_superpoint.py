"""Tests for Superpoint detector descriptor

Authors: Ayush Baid
"""
import unittest

import tests.frontend.detector_descriptor.test_detector_descriptor_base as test_detector_descriptor_base
from gtsfm.frontend.detector.detector_from_joint_detector_descriptor import DetectorFromDetectorDescriptor
from gtsfm.frontend.detector_descriptor.superpoint import SuperPointDetectorDescriptor


class TestSuperPointDetDesc(test_detector_descriptor_base.TestDetectorDescriptorBase):
    """Unit test for Superpoint detector descriptor"""

    def setUp(self):
        """Setup the attributes for the tests."""
        super().setUp()
        self.detector_descriptor = SuperPointDetectorDescriptor(use_cuda=False)

        # explicitly set the detector
        self.detector = DetectorFromDetectorDescriptor(self.detector_descriptor)


if __name__ == "__main__":
    unittest.main()
