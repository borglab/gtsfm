"""Tests for frontend's LMEDS verifier.

Authors: John Lambert
"""
import unittest

import tests.frontend.verifier.test_verifier_base as test_verifier_base
from gtsfm.frontend.verifier.lmeds import LMEDS


class TestLmedsForFundamentalMatrix(test_verifier_base.TestVerifierBase):
    """Unit tests for the LMEDS verifier w/o intrinsics in verification.

    All unit test functions defined in TestVerifierBase are run automatically.
    """

    def setUp(self):
        super().setUp()
        self.verifier = LMEDS(use_intrinsics_in_verification=False, estimation_threshold_px=0.5)


if __name__ == "__main__":
    unittest.main()
