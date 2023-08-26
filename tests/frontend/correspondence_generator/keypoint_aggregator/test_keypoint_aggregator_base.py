"""Unit tests for keypoint aggregation for correspondence generators.

Author: John Lambert
"""
import unittest
from typing import Dict, Tuple

import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_unique import (
    KeypointAggregatorUnique,
)


class TestKeypointAggregatorBase(unittest.TestCase):
    """Main test class for detector base class in frontend."""

    def setUp(self):
        super().setUp()
        self.aggregator = KeypointAggregatorUnique()

    def test_keypoint_aggregator_unique_keypoints(self) -> None:
        """Ensure aggregation works over 3 images, without duplicate keypoints in the same image from separate pairs.

        Image 0 <-> Image 1
          (0,0)  -- (1,1) coordinates

        Image 1 <-> Image 2
          (1,2)  -- (2,2) coordinates

        Image 0 <-> Image 2
          (0,2)  -- (3,3) coordinates
        """
        keypoints_dict: Dict[Tuple[int, int], Keypoints] = {
            (0, 1): (
                Keypoints(coordinates=np.array([[0, 0]], dtype=np.float32)),
                Keypoints(coordinates=np.array([[1, 1]], dtype=np.float32)),
            ),
            (1, 2): (
                Keypoints(coordinates=np.array([[1, 2]], dtype=np.float32)),
                Keypoints(coordinates=np.array([[2, 2]], dtype=np.float32)),
            ),
            (0, 2): (
                Keypoints(coordinates=np.array([[0, 2]], dtype=np.float32)),
                Keypoints(coordinates=np.array([[3, 3]], dtype=np.float32)),
            ),
        }

        keypoints_list, putative_corr_idxs_dict = self.aggregator.aggregate(keypoints_dict)

        assert len(keypoints_list) == 3
        assert all([isinstance(kps, Keypoints) for kps in keypoints_list])

        expected_putative_corr_idxs_dict = {
            (0, 1): np.array([[0, 0]]),
            (1, 2): np.array([[1, 0]]),
            (0, 2): np.array([[1, 1]]),
        }

        assert putative_corr_idxs_dict.keys() == expected_putative_corr_idxs_dict.keys()

        # Should have putative correspondence indices across 3 image pairs.
        assert len(putative_corr_idxs_dict) == 3
        for (i1, i2), putative_corr_idxs in putative_corr_idxs_dict.items():
            assert isinstance(putative_corr_idxs, np.ndarray)
            assert putative_corr_idxs.shape == (1, 2)
            assert np.allclose(putative_corr_idxs_dict[(i1, i2)], expected_putative_corr_idxs_dict[(i1, i2)])

        expected_image0_kps = np.array([[0.0, 0.0], [0.0, 2.0]])
        assert np.allclose(keypoints_list[0].coordinates, expected_image0_kps)

        expected_image1_kps = np.array([[1.0, 1.0], [1.0, 2.0]])
        assert np.allclose(keypoints_list[1].coordinates, expected_image1_kps)

        expected_image2_kps = np.array([[2.0, 2.0], [3.0, 3.0]])
        assert np.allclose(keypoints_list[2].coordinates, expected_image2_kps)


if __name__ == "__main__":
    unittest.main()
