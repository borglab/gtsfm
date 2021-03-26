"""Tests for frontend's RANSAC verifier.

Authors: Ayush Baid
"""
import unittest

import tests.frontend.verifier.test_verifier_base as test_verifier_base
from gtsfm.frontend.verifier.ransac import Ransac


class TestRansac(test_verifier_base.TestVerifierBase):
    """Unit tests for the RANSAC verifier.

    All unit test functions defined in TestVerifierBase are run automatically.
    """

    def setUp(self):
        super().setUp()

        self.verifier = Ransac()


if __name__ == "__main__":
    unittest.main()
