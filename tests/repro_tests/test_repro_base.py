"""Template for reproducibility tests.

Authors: Ayush Baid
"""
import abc
from typing import Any

NUM_REPETITIONS = 10


class ReproducibilityTestBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run_once(self) -> Any:
        """Run the function under test once"""

    @abc.abstractmethod
    def assert_results(self, results_a: Any, results_b: Any) -> None:
        """Compare the two results with assertions"""

    def test_repeatability(self) -> None:
        """Tests the repeatability of the function under test."""
        reference_result = self.run_once()

        for _ in range(NUM_REPETITIONS):
            current_result = self.run_once()
            self.assert_results(current_result, reference_result)
