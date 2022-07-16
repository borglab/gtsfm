"""Tests for the the D2Net detector descriptor.

Authors: John Lambert
"""
import unittest

import tests.frontend.detector_descriptor.test_detector_descriptor_base as test_detector_descriptor_base
from gtsfm.frontend.detector.detector_from_joint_detector_descriptor import DetectorFromDetectorDescriptor
from gtsfm.frontend.detector_descriptor.d2net import D2NetDetDesc


class TestD2NetDetDesc(test_detector_descriptor_base.TestDetectorDescriptorBase):
    """Unit test for the D2Net detector descriptor."""

    def setUp(self) -> None:
        """Setup the attributes for the tests."""
        super().setUp()
        self.detector_descriptor = D2NetDetDesc(use_cuda=False)

        # explicitly set the detector
        self.detector = DetectorFromDetectorDescriptor(self.detector_descriptor)


if __name__ == "__main__":
    unittest.main()
