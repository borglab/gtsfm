"""Tests for frontend's DoG detector class.

Authors: Ayush Baid
"""
import unittest

import tests.frontend.detector.test_detector_base as test_detector_base
from gtsfm.frontend.detector.dog import DoG


class TestDoG(test_detector_base.TestDetectorBase):
    """Test class for DoG detector class in frontend.

    All unit test functions defined in TestDetectorBase are run automatically.
    """

    def setUp(self):
        super().setUp()
        self.detector = DoG()


if __name__ == "__main__":
    unittest.main()
