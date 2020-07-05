""" 
Tests for frontend's FAST detector class.

Authors: Ayush Baid
"""

import tests.frontend.detector.test_detector_base as test_detector_base
from frontend.detector.fast import Fast


class TestFast(test_detector_base.TestDetectorBase):
    """
    Test class for FAST detector class in frontend.

    Note: importing class in this way prevents duplicate runs of the test in the base class
    """

    def setUp(self):
        super().setUp()
        self.detector = Fast()
