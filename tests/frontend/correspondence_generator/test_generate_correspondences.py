"""Tests that plain (no-client) correspondence generation matches the futures path exactly.

The per-cluster frontend runs ``generate_correspondences`` inside a Dask task instead of submitting
nested tasks through ``worker_client()``. These tests pin the two entry points to the same result for the
two generator families that run in that frontend, using deterministic stubs whose outputs depend on the
image shapes (so a shape that is not threaded through correctly shows up as a mismatch).

Authors: Kathirvel Gounder
"""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import pytest
from dask.distributed import Client, LocalCluster

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
import gtsfm.frontend.correspondence_generator.det_desc_correspondence_generator as det_desc_module
from gtsfm.frontend.correspondence_generator.det_desc_correspondence_generator import DetDescCorrespondenceGenerator
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator


class _StubDetectorDescriptor:
    """Deterministic detector-descriptor: keypoint count and layout follow the image size."""

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        num = 3 + image.width % 4
        coords = np.stack([np.arange(num) * image.width / num, np.arange(num) * image.height / num], axis=1)
        coords = coords.astype(np.float32)
        return Keypoints(coordinates=coords), coords / 100.0


class _StubMatcher:
    """Deterministic descriptor matcher whose match count depends on the image shapes it is handed."""

    def match(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, ...],
        im_shape_i2: Tuple[int, ...],
    ) -> np.ndarray:
        num = min(len(keypoints_i1), len(keypoints_i2), 1 + (im_shape_i1[0] + im_shape_i2[1]) % 3)
        return np.stack([np.arange(num), np.arange(num)], axis=1).astype(np.int32)


class _StubImageMatcher:
    """Deterministic image matcher (detector-free): keypoints follow the pair's image sizes."""

    def match(self, image_i1: Image, image_i2: Image) -> Tuple[Keypoints, Keypoints]:
        num = 2 + (image_i1.width + image_i2.width) % 3
        offsets = np.arange(num, dtype=np.float32)[:, None]
        kp1 = np.array([[image_i1.width / 2, image_i1.height / 2]], dtype=np.float32) + offsets
        kp2 = np.array([[image_i2.width / 3, image_i2.height / 3]], dtype=np.float32) + offsets
        return Keypoints(coordinates=kp1), Keypoints(coordinates=kp2)


def _make_images(num_images: int = 5) -> List[Image]:
    return [Image(value_array=np.zeros((40 + 7 * i, 60 + 5 * i, 3), dtype=np.uint8)) for i in range(num_images)]


def test_cpu_matching_remains_parallel_while_geometry_guard_is_held():
    from gtsfm.cluster_optimizer.cluster_vggt import _VGGT_PIPELINE_LOCK

    # All four matches must enter together, even while another task owns the GPU guard.
    ready = threading.Barrier(4)

    class ParallelMatcher(_StubMatcher):
        def match(self, *args, **kwargs):
            ready.wait(timeout=5)
            return super().match(*args, **kwargs)

    generator = DetDescCorrespondenceGenerator(
        detector_descriptor=_StubDetectorDescriptor(), matcher=ParallelMatcher()
    )
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2)]
    with ThreadPoolExecutor(max_workers=1) as executor:
        with _VGGT_PIPELINE_LOCK:
            result = executor.submit(generator.generate_correspondences, _make_images(4), pairs, 4)
            _, matches = result.result(timeout=10)
    assert list(matches) == pairs


def _all_pairs(num_images: int) -> List[Tuple[int, int]]:
    return [(i1, i2) for i1 in range(num_images) for i2 in range(i1 + 1, num_images)]


class TestGenerateCorrespondences(unittest.TestCase):
    """The plain entry point must reproduce the futures entry point exactly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=False, dashboard_address=":0")
        cls.client = Client(cls.cluster)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        cls.cluster.close()

    def _assert_same(
        self,
        expected: Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]],
        actual: Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]],
    ) -> None:
        expected_keypoints, expected_corr = expected
        actual_keypoints, actual_corr = actual
        self.assertEqual(len(expected_keypoints), len(actual_keypoints))
        for kp_expected, kp_actual in zip(expected_keypoints, actual_keypoints):
            np.testing.assert_array_equal(kp_expected.coordinates, kp_actual.coordinates)
        self.assertEqual(set(expected_corr.keys()), set(actual_corr.keys()))
        for edge in expected_corr:
            np.testing.assert_array_equal(expected_corr[edge], actual_corr[edge])

    def test_det_desc_matches_futures(self) -> None:
        generator = DetDescCorrespondenceGenerator(
            matcher=_StubMatcher(), detector_descriptor=_StubDetectorDescriptor()
        )
        images = _make_images()
        visibility_graph = _all_pairs(len(images))

        expected = generator.generate_correspondences_futures(
            self.client, self.client.scatter(images), visibility_graph
        )
        actual = generator.generate_correspondences(images, visibility_graph)

        self._assert_same(expected, actual)
        self.assertEqual(len(actual[1]), len(visibility_graph))
        threaded = generator.generate_correspondences(images, visibility_graph, num_matching_threads=3)
        self._assert_same(expected, threaded)
        self.assertEqual(list(threaded[1]), visibility_graph)

    def test_image_matches_futures(self) -> None:
        generator = ImageCorrespondenceGenerator(matcher=_StubImageMatcher(), deduplicate=True)
        images = _make_images()
        visibility_graph = _all_pairs(len(images))

        expected = generator.generate_correspondences_futures(
            self.client, self.client.scatter(images), visibility_graph
        )
        actual = generator.generate_correspondences(images, visibility_graph)

        self._assert_same(expected, actual)
        self.assertEqual(len(actual[1]), len(visibility_graph))

    def test_empty_visibility_graph(self) -> None:
        generator = DetDescCorrespondenceGenerator(
            matcher=_StubMatcher(), detector_descriptor=_StubDetectorDescriptor()
        )
        images = _make_images(3)
        keypoints_list, corr = generator.generate_correspondences(images, [])
        self.assertEqual(len(keypoints_list), 3)
        self.assertEqual(corr, {})


def test_matching_logs_progress_while_pair_is_running(monkeypatch):
    """A slow first pair must still produce a heartbeat before its result is ready."""
    monkeypatch.setattr(det_desc_module, "MATCHING_PROGRESS_INTERVAL_SECONDS", 0.01)
    progress_seen = threading.Event()

    class WaitingMatcher(_StubMatcher):
        def match(self, *args, **kwargs):
            assert progress_seen.wait(timeout=5), "No progress log while the pair was running"
            return super().match(*args, **kwargs)

    def log_progress(message, *args):
        if "matching %d/%d pairs" in message and args[0] == 0:
            progress_seen.set()

    generator = DetDescCorrespondenceGenerator(WaitingMatcher(), _StubDetectorDescriptor())
    with patch.object(det_desc_module.logger, "info", side_effect=log_progress) as log:
        _, results = generator.generate_correspondences(_make_images(2), [(0, 1)], num_matching_threads=8)
    assert progress_seen.is_set()
    assert list(results) == [(0, 1)]
    assert any("matching %d/%d pairs" in call.args[0] and call.args[1:3] == (1, 1) for call in log.call_args_list)


def test_matching_propagates_errors_and_rejects_invalid_thread_count():
    class FailingMatcher(_StubMatcher):
        def match(self, *args, **kwargs):
            raise RuntimeError("Pair matching failed")

    generator = DetDescCorrespondenceGenerator(FailingMatcher(), _StubDetectorDescriptor())
    with pytest.raises(ValueError, match="at least 1"):
        generator.generate_correspondences(_make_images(2), [(0, 1)], num_matching_threads=0)
    with pytest.raises(RuntimeError, match="Pair matching failed"):
        generator.generate_correspondences(_make_images(2), [(0, 1)], num_matching_threads=2)


def test_recent_matching_rate_excludes_ravenna_cache_warmup():
    """Replay Ravenna's startup: 319 cache hits must not inflate estimates of uncached matching."""
    estimator = det_desc_module._RecentMatchingRate()
    for now, completed in [(5, 320), (10, 328), (15, 336)]:
        assert estimator.update(completed, now) is None
    assert estimator.update(338, 20) == pytest.approx(18 / 15)
    for now, completed in [(25, 345), (30, 351)]:
        estimator.update(completed, now)
    rate = estimator.update(357, 35)
    assert rate == pytest.approx(37 / 30)
    # About 12.5 minutes remain at the observed rate, rather than the old 91-second estimate.
    assert (1281 - 357) / rate == pytest.approx(749.2, abs=0.1)


def test_recent_matching_rate_adapts_to_slowdown_and_stalls():
    estimator = det_desc_module._RecentMatchingRate()
    for now in range(5, 40, 5):
        rate = estimator.update(now * 10, now)
    assert rate == pytest.approx(10)
    for now in range(40, 70, 5):
        rate = estimator.update(350 + (now - 35), now)
    assert rate == pytest.approx(1)
    for now in range(70, 100, 5):
        rate = estimator.update(380, now)
    assert rate == 0  # The caller reports an unknown ETA instead of stale throughput.


if __name__ == "__main__":
    unittest.main()
