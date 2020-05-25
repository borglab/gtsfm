""" Tests for frontend's FAST detector class.

Authors: Ayush Baid
"""

from tests.detector.test_detector_base import TestDetectorBase
from frontend.detector.fast import Fast


class TestFast(TestDetectorBase):
    """test class for FAST detector class in frontend"""

    def setUp(self):
        self.detector = Fast()
