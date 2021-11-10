"""Tests for frontend's RANSAC verifier.

Authors: Ayush Baid
"""
import unittest

import tests.frontend.verifier.test_verifier_base as test_verifier_base
from gtsfm.frontend.verifier.ransac import Ransac


class TestRansacForEssentialMatrix(test_verifier_base.TestVerifierBase):
    """Unit tests for the RANSAC verifier w/ intrinsics in verification.

    All unit test functions defined in TestVerifierBase are run automatically.
    """

    def setUp(self):
        super().setUp()
        self.verifier = Ransac(use_intrinsics_in_verification=True, estimation_threshold_px=0.5)


class TestRansacForFundamentalMatrix(test_verifier_base.TestVerifierBase):
    """Unit tests for the RANSAC verifier w/o intrinsics in verification.

    All unit test functions defined in TestVerifierBase are run automatically.
    """

    def setUp(self):
        super().setUp()
        self.verifier = Ransac(use_intrinsics_in_verification=False, estimation_threshold_px=0.5)


if __name__ == "__main__":
    unittest.main()
