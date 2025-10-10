"""Tests for frontend's keypoint aggregator that de-duplicates within each image detections from each pair.

Authors: John Lambert
"""

import pathlib
import unittest
from typing import Dict, Tuple

import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.cacher.image_matcher_cacher import ImageMatcherCacher
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_dedup import (
    KeypointAggregatorDedup,
)
from gtsfm.frontend.matcher.loftr import LOFTR
from gtsfm.loader.olsson_loader import OlssonLoader
from tests.frontend.correspondence_generator.keypoint_aggregator import test_keypoint_aggregator_base

DATA_ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "data"
DEFAULT_FOLDER = DATA_ROOT_PATH / "set1_lund_door"


class TestKeypointAggregatorDedup(test_keypoint_aggregator_base.TestKeypointAggregatorBase):
    """Test class for DoG detector class in frontend.

    All unit test functions defined in TestDetectorBase are run automatically.
    """

    def setUp(self):
        super().setUp()
        self.aggregator = KeypointAggregatorDedup(nms_merge_radius=0.0)

    def test_keypoint_aggregator_repeated_keypoints(self) -> None:
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
        keypoints_dict: Dict[Tuple[int, int], Tuple[Keypoints, Keypoints]] = {
            (0, 1): (
                Keypoints(coordinates=np.array([[0, 0]], dtype=np.float32)),
                Keypoints(coordinates=np.array([[1, 1]], dtype=np.float32)),
            ),
            (1, 2): (
                Keypoints(coordinates=np.array([[1, 2]], dtype=np.float32)),
                Keypoints(coordinates=np.array([[2, 2]], dtype=np.float32)),
            ),
            (0, 2): (
                Keypoints(coordinates=np.array([[0, 2], [0, 0]], dtype=np.float32)),
                Keypoints(coordinates=np.array([[3, 3], [4, 4]], dtype=np.float32)),
            ),
        }

        keypoints_list, putative_corr_idxs_dict = self.aggregator.aggregate(keypoints_dict)

        assert len(keypoints_list) == 3
        assert all([isinstance(kps, Keypoints) for kps in keypoints_list])

        # Duplicates should have been removed.
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

    def test_dedup_nms_merge_radius_3(self) -> None:
        loader = OlssonLoader(str(DEFAULT_FOLDER), max_frame_lookahead=4)
        image_matcher = ImageMatcherCacher(matcher_obj=LOFTR())

        images = [loader.get_image(i) for i in range(4)]
        image_pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]
        aggregator = KeypointAggregatorDedup(nms_merge_radius=3.0)

        pairwise_correspondences = {
            (i1, i2): image_matcher.match(image_i1=images[i1], image_i2=images[i2]) for i1, i2 in image_pairs
        }

        keypoints_list, putative_corr_idxs_dict = aggregator.aggregate(keypoints_dict=pairwise_correspondences)

        for (i1, i2), corr_idxs in putative_corr_idxs_dict.items():
            assert corr_idxs.dtype == np.int32
            assert len(corr_idxs.shape) == 2
            assert corr_idxs.shape[-1] == 2


if __name__ == "__main__":
    unittest.main()
