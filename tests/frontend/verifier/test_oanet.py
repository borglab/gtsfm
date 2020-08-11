"""
Tests for frontend's base verifier class.

Authors: Ayush Baid
"""

import random
from typing import List, Tuple
from unittest.mock import MagicMock

import dask
import numpy as np

import tests.frontend.verifier.test_verifier_base as test_verifier_base
from frontend.verifier.oanet import OANetVerifier


class TestOANet(test_verifier_base.TestVerifierBase):
    """
    Unit tests for the Base Verifier class.

    Should be inherited by all verifier unit tests.
    """

    def setUp(self):
        super().setUp()

        self.verifier = OANetVerifier()
