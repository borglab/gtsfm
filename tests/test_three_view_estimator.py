"""Unit tests for the three view estimator.

Authors: Ayush Baid
"""
import unittest

import numpy as np

from gtsfm.three_view_estimator import ThreeViewEstimator


class TestThreeViewEstimator(unittest.TestCase):
    """Unit test for three view estimator."""

    def test_match_triplet_using_pairwise_matches(self):
        """Tests formation of triplet matches using three pairwise match indices."""

        match_indices_i1i2 = np.array(
            [
                [0, 4],  # has match in i2i3 as well as i1i3, and they are equal
                [1, 3],  # has match in i2i3 as well as i1i3, and they are not equal
                [4, 7],  # has match in i2i3, but not in i1i3
                [7, 15],  # has match in i1i3, but not in i2i3
            ],
            dtype=np.int64,
        )
        match_indices_i2i3 = np.array([[4, 9], [3, 5], [7, 0], [20, 2]], dtype=np.int64)
        match_indices_i1i3 = np.array([[0, 9], [1, 4], [13, 13], [7, 3]], dtype=np.int64)

        expected = np.array([[0, 4, 9]], dtype=np.int64)

        computed = ThreeViewEstimator.match_triplet_using_pairwise_matches(
            match_indices_i1i2, match_indices_i2i3, match_indices_i1i3
        )
        np.testing.assert_allclose(computed, expected)


if __name__ == "__main__":
    unittest.main()
