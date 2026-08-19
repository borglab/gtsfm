"""Correspondence generator that utilizes explicit keypoint detection, following by descriptor matching, per image.

Authors: John Lambert
"""

import time
from typing import Dict, List, Tuple

import numpy as np
from dask.distributed import Client, Future

import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.products.visibility_graph import VisibilityGraph

logger = logger_utils.get_logger()


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
        visibility_graph: VisibilityGraph,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Apply the correspondence generator to generate putative correspondences.

        Args:
            client: Dask client, used to execute the front-end as futures.
            images: List of all images, as futures.
            visibility_graph: The visibility graph defining which image pairs to process.

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
            **kwargs,
        ) -> np.ndarray:
            return feature_matcher.match(features_i1[0], features_i2[0], features_i1[1], features_i2[1], **kwargs)

        det_desc_future = client.scatter(self._detector_descriptor, broadcast=False)
        features_futures = [client.submit(apply_det_desc, det_desc_future, image) for image in images]
        del det_desc_future  # free memory (on workers)
        feature_matcher_future = client.scatter(self._matcher, broadcast=False)
        image_shapes_futures = [client.submit(get_image_shape, image) for image in images]

        putative_corr_idxs_futures = {
            (i1, i2): client.submit(
                apply_matcher,
                feature_matcher_future,
                features_futures[i1],
                features_futures[i2],
                im_shape_i1=image_shapes_futures[i1],
                im_shape_i2=image_shapes_futures[i2],
            )
            for (i1, i2) in visibility_graph
        }

        putative_corr_idxs_dict = client.gather(putative_corr_idxs_futures)
        keypoints_futures = client.map(lambda f: f[0], features_futures)
        keypoints_list = client.gather(keypoints_futures)

        return keypoints_list, putative_corr_idxs_dict

    def generate_correspondences_inline(
        self,
        images: List[Image],
        visibility_graph: VisibilityGraph,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Inline, no-Dask variant of ``generate_correspondences``.

        Detection runs once per image and matching once per pair, in plain Python loops calling the
        (cache-backed) detector-descriptor and matcher directly. Computed-and-discarded item by item, so
        peak memory is bounded to one cluster's features rather than a worker-resident pile of every
        feature/correspondence future. Mirrors the synchronous style of ``ColmapCorrespondenceGenerator``.

        Args:
            images: Materialized images indexed by position (``images[i]`` is image ``i``).
            visibility_graph: Image pairs ``(i1, i2)`` to match; indices reference ``images``.

        Returns:
            keypoints_list: one ``Keypoints`` per image, in index order.
            putative_corr_idxs_dict: per-pair putative correspondence indices.
        """
        num_images = len(images)
        logger.info("🔵 [frontend] Detecting + describing features on %d images...", num_images)
        det_start = time.time()
        features: List[Tuple[Keypoints, np.ndarray]] = []
        for i, image in enumerate(images):
            features.append(self._detector_descriptor.detect_and_describe(image))
            if (i + 1) % 250 == 0 or (i + 1) == num_images:
                logger.info("🔵 [frontend] detection %d/%d images (%.0fs)", i + 1, num_images, time.time() - det_start)
        keypoints_list = [keypoints for keypoints, _ in features]

        num_pairs = len(visibility_graph)
        logger.info("🔵 [frontend] Matching %d pairs...", num_pairs)
        match_start = time.time()
        putative_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray] = {}
        for p, (i1, i2) in enumerate(visibility_graph):
            keypoints_i1, descriptors_i1 = features[i1]
            keypoints_i2, descriptors_i2 = features[i2]
            putative_corr_idxs_dict[(i1, i2)] = self._matcher.match(
                keypoints_i1,
                keypoints_i2,
                descriptors_i1,
                descriptors_i2,
                im_shape_i1=images[i1].shape,
                im_shape_i2=images[i2].shape,
            )
            if (p + 1) % 5000 == 0 or (p + 1) == num_pairs:
                elapsed = time.time() - match_start
                rate = (p + 1) / elapsed if elapsed > 0 else 0.0
                eta = (num_pairs - (p + 1)) / rate if rate > 0 else 0.0
                logger.info(
                    "🔵 [frontend] matching %d/%d pairs (%.0fs, %.0f pair/s, ETA %.0fs)",
                    p + 1, num_pairs, elapsed, rate, eta,
                )

        return keypoints_list, putative_corr_idxs_dict
