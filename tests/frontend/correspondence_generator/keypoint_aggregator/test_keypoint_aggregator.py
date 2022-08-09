"""Unit tests for keypoint aggregation for correspondence generators.

Author: John Lambert
"""
from typing import Dict, Tuple, Union

import numpy as np
import pytest

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_dedup import (
    KeypointAggregatorDedup,
)
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_unique import (
    KeypointAggregatorUnique,
)


@pytest.mark.parametrize("aggregator", [KeypointAggregatorUnique(), KeypointAggregatorDedup()])
def test_keypoint_aggregator_unique_keypoints(
    aggregator: Union[KeypointAggregatorDedup, KeypointAggregatorUnique]
) -> None:
    """Ensure aggregation works over 3 images, without duplicate keypoints in the same image from separate pairs.

    Image 0 <-> Image 1
      (0,0)  -- (1,1) coordinates

    Image 1 <-> Image 2
      (1,2)  -- (2,2) coordinates

    Image 0 <-> Image 2
      (0,2)  -- (3,3) coordinates
    """
    keypoints_dict: Dict[Tuple[int, int], Keypoints] = {
        (0, 1): (Keypoints(coordinates=np.array([[0, 0]])), Keypoints(coordinates=np.array([[1, 1]]))),
        (1, 2): (Keypoints(coordinates=np.array([[1, 2]])), Keypoints(coordinates=np.array([[2, 2]]))),
        (0, 2): (Keypoints(coordinates=np.array([[0, 2]])), Keypoints(coordinates=np.array([[3, 3]]))),
    }

    keypoints_list, putative_corr_idxs_dict = aggregator.run(keypoints_dict)

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


def test_keypoint_aggregator_repeated_keypoints() -> None:
    """Ensure aggregation works over 3 images, with duplicate keypoints in the same image from separate pairs.

    Create tracks.

    Image 0 <-> Image 1
      (0,0)  -- (1,1)

    Image 1 <-> Image 2
      (1,2)  -- (2,2)

    Image 0 <-> Image 2
      (0,2)  -- (3,3)
      (0,0)  -- (4,4)
    """
    keypoints_dict: Dict[Tuple[int, int], Keypoints] = {
        (0, 1): (Keypoints(coordinates=np.array([[0, 0]])), Keypoints(coordinates=np.array([[1, 1]]))),
        (1, 2): (Keypoints(coordinates=np.array([[1, 2]])), Keypoints(coordinates=np.array([[2, 2]]))),
        (0, 2): (Keypoints(coordinates=np.array([[0, 2], [0, 0]])), Keypoints(coordinates=np.array([[3, 3], [4, 4]]))),
    }

    aggregator = KeypointAggregatorDedup()
    keypoints_list, putative_corr_idxs_dict = aggregator.run(keypoints_dict)

    assert len(keypoints_list) == 3
    assert all([isinstance(kps, Keypoints) for kps in keypoints_list])

    # removing duplicates
    expected_putative_corr_idxs_dict = {
        (0, 1): np.array([[0, 0]]),
        (1, 2): np.array([[1, 0]]),
        (0, 2): np.array([[1, 1], [0, 2]]),
    }

    assert putative_corr_idxs_dict.keys() == expected_putative_corr_idxs_dict.keys()

    assert len(putative_corr_idxs_dict) == 3
    for (i1, i2), putative_corr_idxs in putative_corr_idxs_dict.items():
        assert isinstance(putative_corr_idxs, np.ndarray)
        assert putative_corr_idxs.shape[1] == 2
        assert np.allclose(putative_corr_idxs_dict[(i1, i2)], expected_putative_corr_idxs_dict[(i1, i2)])

    # with de-dup
    expected_image0_kps = np.array([[0.0, 0.0], [0.0, 2.0]])
    assert np.allclose(keypoints_list[0].coordinates, expected_image0_kps)

    expected_image1_kps = np.array([[1.0, 1.0], [1.0, 2.0]])
    assert np.allclose(keypoints_list[1].coordinates, expected_image1_kps)

    expected_image2_kps = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    assert np.allclose(keypoints_list[2].coordinates, expected_image2_kps)
