"""Tests for frontend's FAST detector class.

Authors: Ayush Baid
"""
import unittest

import tests.frontend.detector.test_detector_base as test_detector_base
from gtsfm.frontend.detector.fast import Fast


class TestFast(test_detector_base.TestDetectorBase):
    """Test class for FAST detector class in frontend.

    All unit test functions defined in TestDetectorBase are run automatically.
    """

    def setUp(self):
        super().setUp()
        self.detector = Fast()


if __name__ == "__main__":
    unittest.main()
