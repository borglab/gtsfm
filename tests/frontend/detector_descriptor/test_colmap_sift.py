"""Tests for SIFT detector descriptor

Authors: Ayush Baid
"""
import unittest

import tests.frontend.detector_descriptor.test_detector_descriptor_base as test_detector_descriptor_base
from gtsfm.frontend.detector.detector_from_joint_detector_descriptor import DetectorFromDetectorDescriptor
from gtsfm.frontend.detector_descriptor.colmap_sift import ColmapSIFTDetectorDescriptor


class TestColmapSIFTDetectorDescriptor(test_detector_descriptor_base.TestDetectorDescriptorBase):
    """Main test class for detector-description combination base class in frontend.

    All unit test functions defined in TestDetectorDescriptorBase are run automatically.
    """

    def setUp(self):
        """Setup the attributes for the tests."""
        super().setUp()
        # Note: pycolmap does not guarantee that the number of keypoints will not exceed the specified maximum, as there
        # can be ties in terms of the response scores. E.g., if the 5000th keypoint and the 5001st keypoint have the
        # same response, pycolmap will return 5001 keypoints. Setting the number of maximum keypoints lower reduces the
        # risk of this happening.
        self.detector_descriptor = ColmapSIFTDetectorDescriptor(max_keypoints=2000)

        # explicitly set the detector
        self.detector = DetectorFromDetectorDescriptor(self.detector_descriptor)


if __name__ == "__main__":
    unittest.main()
