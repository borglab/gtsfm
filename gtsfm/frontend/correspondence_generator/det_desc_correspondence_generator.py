"""Correspondence generator that utilizes explicit keypoint detection, following by descriptor matching, per image.

Authors: John Lambert
"""

import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
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
MATCHING_PROGRESS_INTERVAL_SECONDS = 5.0


class _RecentMatchingRate:
    """Estimate throughput over recent progress reports, excluding the initial cache-loading burst."""

    def __init__(self) -> None:
        self._samples: deque[Tuple[float, int]] = deque()

    def update(self, completed: int, now: float) -> float | None:
        # The first report establishes a baseline: do not count its startup/cache-hit completions.
        self._samples.append((now, completed))
        while len(self._samples) > 1 and self._samples[1][0] <= now - 30.0:
            self._samples.popleft()
        start_time, start_completed = self._samples[0]
        elapsed = now - start_time
        # Several reporting intervals smooth out batches finishing together across matching threads.
        if elapsed < 15.0:
            return None
        return (completed - start_completed) / elapsed


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
        num_matching_threads: int = 1,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Generate putative correspondences in the calling process (no Dask client).

        Same per-image detection and per-pair matching kernels as ``generate_correspondences_futures``,
        with sequential detection and a bounded thread pool for matching. This is the cluster-frontend
        entry point (that code already executes inside a Dask task), and peak memory stays bounded to
        one cluster's features plus one active match per thread.

        Args:
            images: Materialized images indexed by position (``images[i]`` is image ``i``).
            visibility_graph: Image pairs ``(i1, i2)`` to match; indices reference ``images``.
            num_matching_threads: Concurrent pair matches. Defaults to one to avoid nested parallelism
                when multiple cluster frontends already execute on Dask workers.

        Returns:
            keypoints_list: one ``Keypoints`` per image, in index order.
            putative_corr_idxs_dict: per-pair putative correspondence indices.
        """
        if num_matching_threads < 1:
            raise ValueError("num_matching_threads must be at least 1")
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
        if num_pairs == 0:
            return keypoints_list, {}
        num_threads = min(num_matching_threads, num_pairs)
        logger.info("🔵 [frontend] Matching %d pairs with %d threads...", num_pairs, num_threads)
        match_start = time.monotonic()
        last_progress = match_start
        recent_rate = _RecentMatchingRate()
        putative_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray] = {}
        pairs = iter(visibility_graph)
        with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="gtsfm-match") as executor:

            def submit_pair(pair: Tuple[int, int]):
                i1, i2 = pair
                return executor.submit(
                    _match_features, self._matcher, features[i1], features[i2], images[i1].shape, images[i2].shape
                )

            # Keep only one pending match per thread, rather than queueing the entire graph.
            pending = {submit_pair(pair): pair for pair in islice(pairs, num_threads)}
            while pending:
                timeout = max(0.0, MATCHING_PROGRESS_INTERVAL_SECONDS - (time.monotonic() - last_progress))
                done, _ = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                for future in done:
                    pair = pending.pop(future)
                    putative_corr_idxs_dict[pair] = future.result()
                for _ in done:
                    pair = next(pairs, None)
                    if pair is not None:
                        pending[submit_pair(pair)] = pair
                now = time.monotonic()
                if now - last_progress >= MATCHING_PROGRESS_INTERVAL_SECONDS or not pending:
                    completed = len(putative_corr_idxs_dict)
                    elapsed = now - match_start
                    rate = recent_rate.update(completed, now)
                    rate_label = f"{rate:.2f} pair/s" if rate is not None else "estimating"
                    if not pending:
                        eta = "0s"
                    elif rate is None:
                        eta = "estimating"
                    else:
                        eta = f"{(num_pairs - completed) / rate:.0f}s" if rate > 0 else "unknown"
                    logger.info(
                        "🔵 [frontend] matching %d/%d pairs (%.0fs, recent rate %s, ETA %s)",
                        completed,
                        num_pairs,
                        elapsed,
                        rate_label,
                        eta,
                    )
                    last_progress = now

        # Preserve input ordering regardless of which matches finish first.
        return keypoints_list, {pair: putative_corr_idxs_dict[pair] for pair in visibility_graph}
