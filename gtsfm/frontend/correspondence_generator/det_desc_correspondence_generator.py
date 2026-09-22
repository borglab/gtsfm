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


def _detect_and_describe(detector_descriptor: DetectorDescriptorBase, image: Image) -> Tuple[Keypoints, np.ndarray]:
    """Per-image kernel: detect keypoints and compute their descriptors."""
    return detector_descriptor.detect_and_describe(image)


def _image_shape(image: Image) -> Tuple[int, ...]:
    return image.shape


def _match_features(
    matcher: MatcherBase,
    features_i1: Tuple[Keypoints, np.ndarray],
    features_i2: Tuple[Keypoints, np.ndarray],
    im_shape_i1: Tuple[int, ...],
    im_shape_i2: Tuple[int, ...],
) -> np.ndarray:
    """Per-pair kernel: match two images' (keypoints, descriptors)."""
    return matcher.match(
        features_i1[0],
        features_i2[0],
        features_i1[1],
        features_i2[1],
        im_shape_i1=im_shape_i1,
        im_shape_i2=im_shape_i2,
    )


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

    def generate_correspondences_futures(
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
        det_desc_future = client.scatter(self._detector_descriptor, broadcast=False)
        features_futures = [client.submit(_detect_and_describe, det_desc_future, image) for image in images]
        del det_desc_future  # free memory (on workers)
        feature_matcher_future = client.scatter(self._matcher, broadcast=False)
        image_shapes_futures = [client.submit(_image_shape, image) for image in images]

        putative_corr_idxs_futures = {
            (i1, i2): client.submit(
                _match_features,
                feature_matcher_future,
                features_futures[i1],
                features_futures[i2],
                image_shapes_futures[i1],
                image_shapes_futures[i2],
            )
            for (i1, i2) in visibility_graph
        }

        putative_corr_idxs_dict = client.gather(putative_corr_idxs_futures)
        keypoints_futures = client.map(lambda f: f[0], features_futures)
        keypoints_list = client.gather(keypoints_futures)

        return keypoints_list, putative_corr_idxs_dict

    def generate_correspondences(
        self,
        images: List[Image],
        visibility_graph: VisibilityGraph,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Generate putative correspondences in the calling process (no Dask client).

        Same per-image detection and per-pair matching kernels as ``generate_correspondences_futures``,
        run in plain loops. This is the cluster-frontend entry point (that code already executes inside a
        Dask task), and peak memory stays bounded to one cluster's features rather than a worker-resident
        pile of every feature/correspondence future.

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
            features.append(_detect_and_describe(self._detector_descriptor, image))
            if (i + 1) % 250 == 0 or (i + 1) == num_images:
                logger.info("🔵 [frontend] detection %d/%d images (%.0fs)", i + 1, num_images, time.time() - det_start)
        keypoints_list = [keypoints for keypoints, _ in features]

        num_pairs = len(visibility_graph)
        logger.info("🔵 [frontend] Matching %d pairs...", num_pairs)
        match_start = time.time()
        putative_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray] = {}
        for p, (i1, i2) in enumerate(visibility_graph):
            putative_corr_idxs_dict[(i1, i2)] = _match_features(
                self._matcher, features[i1], features[i2], images[i1].shape, images[i2].shape
            )
            if (p + 1) % 5000 == 0 or (p + 1) == num_pairs:
                elapsed = time.time() - match_start
                rate = (p + 1) / elapsed if elapsed > 0 else 0.0
                eta = (num_pairs - (p + 1)) / rate if rate > 0 else 0.0
                logger.info(
                    "🔵 [frontend] matching %d/%d pairs (%.0fs, %.0f pair/s, ETA %.0fs)",
                    p + 1,
                    num_pairs,
                    elapsed,
                    rate,
                    eta,
                )

        return keypoints_list, putative_corr_idxs_dict
