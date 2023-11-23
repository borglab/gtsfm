"""Template for reproducibility tests.

Authors: Ayush Baid
"""
import abc
from typing import Any, List

NUM_REPETITIONS = 10


class ReproducibilityTestBase(metaclass=abc.ABCMeta):
    """A base class to define reproducibility tests.

    The class provides two ways to test reproducibility:
        - assert_results(): To compare the result at each iteration to a reference.
        - assert_results_statistics() (Optional): To check statistics on all results together.
    """

    @abc.abstractmethod
    def run_once(self) -> Any:
        """Run the function under test once"""

    @abc.abstractmethod
    def assert_results(self, results_a: Any, results_b: Any) -> None:
        """Compare the two results with assertions"""

    def test_repeatability(self) -> None:
        """Tests the repeatability of the function under test."""
        reference_result = self.run_once()
        all_results = [reference_result]

        for _ in range(NUM_REPETITIONS):
            current_result = self.run_once()
            all_results.append(current_result)
            self.assert_results(current_result, reference_result)

        self.assert_results_statistics(all_results)

    def assert_results_statistics(self, all_results: List[Any]) -> None:
        """Check for any statistics across all_results.

        This default implementation does not check anything, but can be overrided by derived classes."""
        return None
