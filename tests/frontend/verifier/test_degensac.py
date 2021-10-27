"""Tests for frontend's Degensac verifier.

Authors: Ayush Baid
"""
import unittest

import tests.frontend.verifier.test_verifier_base as test_verifier_base
from gtsfm.frontend.verifier.degensac import Degensac


class TestDegensac(test_verifier_base.TestVerifierBase):
    """Unit tests for the Degensac verifier.

    All unit test functions defined in TestVerifierBase are run automatically.
    """

    def setUp(self):
        super().setUp()

        self.verifier = Degensac(use_intrinsics_in_verification=False, estimation_threshold_px=0.5)


if __name__ == "__main__":
    unittest.main()
