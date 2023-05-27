"""Tests for frontend's base detector-descriptor class.

Authors: Ayush Baid
"""
import unittest

import numpy as np

import tests.frontend.detector.test_detector_base as test_detector_base
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.descriptor.dummy_descriptor import DummyDescriptor
from gtsfm.frontend.detector.detector_from_joint_detector_descriptor import DetectorFromDetectorDescriptor
from gtsfm.frontend.detector.dummy_detector import DummyDetector
from gtsfm.frontend.detector_descriptor.combination_detector_descriptor import CombinationDetectorDescriptor


class TestDetectorDescriptorBase(test_detector_base.TestDetectorBase):
    """Main test class for detector-description combination base class in frontend.

    We re-use detector specific test cases from TestDetectorBase
    """

    def setUp(self):
        """Setup the attributes for the tests."""
        super().setUp()
        self.detector_descriptor = CombinationDetectorDescriptor(DummyDetector(), DummyDescriptor())

        # explicitly set the detector
        self.detector = DetectorFromDetectorDescriptor(self.detector_descriptor)

    def test_detect_and_describe_shape(self):
        """Tests that the number of keypoints and descriptors are the same."""

        # test on random indexes
        test_indices = [0, 5]
        for idx in test_indices:
            kps, descs = self.detector_descriptor.apply(self.loader.get_image(idx))

            if len(kps) == 0:
                # test-case for empty results
                self.assertEqual(0, descs.size)
            else:
                # number of descriptors and features should be equal
                self.assertEqual(len(kps), descs.shape[0])


if __name__ == "__main__":
    unittest.main()
