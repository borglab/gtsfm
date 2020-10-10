"""
Tests for frontend's DoG detector class.

Authors: Ayush Baid
"""

import tests.frontend.detector.test_detector_base as test_detector_base
from frontend.detector.dog import DoG


class TestDoG(test_detector_base.TestDetectorBase):
    """Test class for DoG detector class in frontend."""

    def setUp(self):
        super().setUp()
        self.detector = DoG()
