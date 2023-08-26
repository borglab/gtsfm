"""Correspondence generator that utilizes explicit keypoint detection, following by descriptor matching, per image.

Authors: John Lambert
"""
from typing import Dict, List, Tuple

import numpy as np
from dask.distributed import Client, Future

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase
from gtsfm.frontend.matcher.matcher_base import MatcherBase


class DetDescCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Traditional pair-wise matching of descriptors."""

    def __init__(self, matcher: MatcherBase, detector_descriptor: DetectorDescriptorBase) -> None:
        self._detector_descriptor = detector_descriptor
        self._matcher = matcher

    def __repr__(self) -> str:
        return f"""
        DetDescCorrespondenceGenerator:
           {self._detector_descriptor}
           {self._matcher}
        """

    def generate_correspondences(
        self,
        client: Client,
        images: List[Future],
        image_pairs: List[Tuple[int, int]],
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Apply the correspondence generator to generate putative correspondences.

        Args:
            client: dask client, used to execute the front-end as futures.
            images: list of all images, as futures.
            image_pairs: indices of the pairs of images to estimate two-view pose and correspondences.

        Returns:
            List of keypoints, one entry for each input images.
            Putative correspondence as indices of keypoints, for pairs of images.
        """

        def apply_det_desc(det_desc: DetectorDescriptorBase, image: Image) -> Tuple[Keypoints, np.ndarray]:
            return det_desc.detect_and_describe(image)

        def get_image_shape(image: Image) -> Tuple[int, int, int]:
            return image.shape

        def apply_matcher(
            feature_matcher: MatcherBase,
            features_i1: Tuple[Keypoints, np.ndarray],
            features_i2: Tuple[Keypoints, np.ndarray],
            im_shape_i1: Tuple[int, int, int],
            im_shape_i2: Tuple[int, int, int],
        ) -> np.ndarray:
            return feature_matcher.match(
                features_i1[0], features_i2[0], features_i1[1], features_i2[1], im_shape_i1, im_shape_i2
            )

        det_desc_future = client.scatter(self._detector_descriptor, broadcast=False)
        feature_matcher_future = client.scatter(self._matcher, broadcast=False)
        features_futures = [client.submit(apply_det_desc, det_desc_future, image) for image in images]
        image_shapes_futures = [client.submit(get_image_shape, image) for image in images]

        putative_corr_idxs_futures = {
            (i1, i2): client.submit(
                apply_matcher,
                feature_matcher_future,
                features_futures[i1],
                features_futures[i2],
                image_shapes_futures[i1],
                image_shapes_futures[i2],
            )
            for (i1, i2) in image_pairs
        }

        putative_corr_idxs_dict = client.gather(putative_corr_idxs_futures)
        keypoints_futures = client.map(lambda f: f[0], features_futures)
        keypoints_list = client.gather(keypoints_futures)

        return keypoints_list, putative_corr_idxs_dict
