"""Tests for frontend's base verifier class.

Authors: Ayush Baid
"""
import unittest

import tests.frontend.verifier.test_verifier_base as test_verifier_base
from frontend.verifier.degensac import Degensac


class TestDegensac(test_verifier_base.TestVerifierBase):
    """Unit tests for the Degensac verifier.

    All unit test functions defined in TestVerifierBase are run automatically.
    """

    def setUp(self):
        super().setUp()

        self.verifier = Degensac()


if __name__ == "__main__":
    unittest.main()
