"""Tests that inline (no-Dask) correspondence generation matches the Dask path exactly.

The per-cluster frontend runs ``generate_correspondences_inline`` inside a Dask task instead of submitting
nested tasks through ``worker_client()``. These tests pin the two entry points to the same result for the
two generator families that run in that frontend, using deterministic stubs whose outputs depend on the
image shapes (so a shape that is not threaded through correctly shows up as a mismatch).

Authors: Kathirvel Gounder
"""

import unittest
from typing import Dict, List, Tuple

import numpy as np
from dask.distributed import Client, LocalCluster

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
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


def _all_pairs(num_images: int) -> List[Tuple[int, int]]:
    return [(i1, i2) for i1 in range(num_images) for i2 in range(i1 + 1, num_images)]


class TestGenerateCorrespondencesInline(unittest.TestCase):
    """The inline entry point must reproduce the Dask entry point exactly."""

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

    def test_det_desc_inline_matches_dask(self) -> None:
        generator = DetDescCorrespondenceGenerator(
            matcher=_StubMatcher(), detector_descriptor=_StubDetectorDescriptor()
        )
        images = _make_images()
        visibility_graph = _all_pairs(len(images))

        expected = generator.generate_correspondences(self.client, self.client.scatter(images), visibility_graph)
        actual = generator.generate_correspondences_inline(images, visibility_graph)

        self._assert_same(expected, actual)
        self.assertEqual(len(actual[1]), len(visibility_graph))

    def test_image_inline_matches_dask(self) -> None:
        generator = ImageCorrespondenceGenerator(matcher=_StubImageMatcher(), deduplicate=True)
        images = _make_images()
        visibility_graph = _all_pairs(len(images))

        expected = generator.generate_correspondences(self.client, self.client.scatter(images), visibility_graph)
        actual = generator.generate_correspondences_inline(images, visibility_graph)

        self._assert_same(expected, actual)
        self.assertEqual(len(actual[1]), len(visibility_graph))

    def test_empty_visibility_graph(self) -> None:
        generator = DetDescCorrespondenceGenerator(
            matcher=_StubMatcher(), detector_descriptor=_StubDetectorDescriptor()
        )
        images = _make_images(3)
        keypoints_list, corr = generator.generate_correspondences_inline(images, [])
        self.assertEqual(len(keypoints_list), 3)
        self.assertEqual(corr, {})


if __name__ == "__main__":
    unittest.main()
