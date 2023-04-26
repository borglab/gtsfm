"""Correspondence generator that utilizes explicit keypoint detection, following by descriptor matching, per image.

Authors: John Lambert
"""

from typing import Dict, List, Tuple

from dask.delayed import Delayed

from gtsfm.feature_extractor import FeatureExtractor
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.matcher.matcher_base import MatcherBase


class DetDescCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Traditional pair-wise matching of descriptors."""

    def __init__(self, matcher: MatcherBase, feature_extractor: FeatureExtractor):
        """
        Args:
            matcher: matcher to use.
            feature_extractor: feature extractor to use.
        """
        self._feature_extractor = feature_extractor
        self._matcher = matcher

    def create_computation_graph(
        self,
        delayed_images: List[Delayed],
        delayed_image_shapes: List[Delayed],
        image_pair_indices: List[Tuple[int, int]],
    ) -> Tuple[List[Delayed], Dict[Tuple[int, int], Delayed]]:
        """Create Dask computation graph for correspondence generation.

        Args:
            delayed_images: list of N images.
            delayed_image_shapes: list of N image shapes, as tuples (H,W) in pixels, wrapped as Delayed.
            image_pair_indices: list of image pairs, each represented by a tuple (i1,i2).

        Return:
            delayed_keypoints: list of delayed tasks, each yielding Keypoints in one image.
            delayed_putative_corr_idxs_dict: mapping from image pair (i1,i2) to delayed task to compute
                putative correspondence indices. Correspondence indices are represented by an array of
                shape (K,2), for K correspondences.
        """
        # Detection and description graph.
        delayed_keypoints = []
        delayed_descriptors = []
        for delayed_image in delayed_images:
            (delayed_dets, delayed_descs) = self._feature_extractor.create_computation_graph(delayed_image)
            delayed_keypoints.append(delayed_dets)
            delayed_descriptors.append(delayed_descs)

        delayed_putative_corr_idxs_dict: Dict[Tuple[int, int], Delayed] = {}
        for (i1, i2) in image_pair_indices:
            # Graph for matching to obtain putative correspondences.
            delayed_putative_corr_idxs_dict[i1, i2] = self._matcher.create_computation_graph(
                keypoints_i1_graph=delayed_keypoints[i1],
                keypoints_i2_graph=delayed_keypoints[i2],
                descriptors_i1_graph=delayed_descriptors[i1],
                descriptors_i2_graph=delayed_descriptors[i2],
                im_shape_i1=delayed_image_shapes[i1],
                im_shape_i2=delayed_image_shapes[i2],
            )

        return delayed_keypoints, delayed_putative_corr_idxs_dict
