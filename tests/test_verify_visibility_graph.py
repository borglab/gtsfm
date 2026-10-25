"""Tests for the global two-view verification pass (``scene_optimizer.verify_visibility_graph``).

The pass dispatches on worker-pool size: one worker runs the frontend inline in the calling process,
several workers fan it out and gather only the lean v_corr sub-dicts. These tests pin both arms to the
same verified graph and correspondences, using deterministic stubs (an edge is verified iff its index
sum is even).

Authors: Kathirvel Gounder
"""

import unittest
from typing import List, Tuple

import numpy as np
from dask.distributed import Client, LocalCluster

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.correspondence_generator.det_desc_correspondence_generator import DetDescCorrespondenceGenerator
from gtsfm.scene_optimizer import verify_visibility_graph


class _StubDetectorDescriptor:
    """Deterministic detector-descriptor: keypoint count and layout follow the image size."""

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        num = 4 + image.width % 3
        coords = np.stack([np.arange(num) * image.width / num, np.arange(num) * image.height / num], axis=1)
        coords = coords.astype(np.float32)
        return Keypoints(coordinates=coords), coords / 100.0


class _StubMatcher:
    """Deterministic descriptor matcher: pairs the first min(n1, n2) keypoints."""

    def match(self, keypoints_i1, keypoints_i2, descriptors_i1, descriptors_i2, im_shape_i1, im_shape_i2):
        num = min(len(keypoints_i1), len(keypoints_i2))
        return np.stack([np.arange(num), np.arange(num)], axis=1).astype(np.int32)


class _StubResult:
    def __init__(self, v_corr_idxs: np.ndarray, is_valid: bool) -> None:
        self.v_corr_idxs = v_corr_idxs
        self._is_valid = is_valid

    def valid(self) -> bool:
        return self._is_valid


class _StubTwoViewEstimator:
    """Verifies an edge iff its index sum is even; v_corr derives from the pair so edges are distinct."""

    def run_2view(self, *, putative_corr_idxs, i1, i2, **kwargs) -> _StubResult:
        return _StubResult(putative_corr_idxs + (i1 * 1000 + i2), (i1 + i2) % 2 == 0)


class _StubOneViewData:
    """Per-image data with the attributes the two-view loop reads (unused by the stub estimator)."""

    def __init__(self) -> None:
        self.intrinsics = None
        self.camera_gt = None


class _StubLoader:
    """Minimal loader surface consumed by the verification pass."""

    def __init__(self, num_images: int) -> None:
        self._num_images = num_images

    def __len__(self) -> int:
        return self._num_images

    def get_relative_pose_priors(self, pairs) -> dict:
        return {}

    def get_gt_scene_trimesh(self):
        return None


def _make_images(num_images: int) -> List[Image]:
    return [Image(value_array=np.zeros((30 + 5 * i, 45 + 7 * i, 3), dtype=np.uint8)) for i in range(num_images)]


def _run_with_workers(n_workers: int, num_images: int = 6):
    """Run the verification pass on a fresh local cluster with the given worker count."""
    images = _make_images(num_images)
    visibility_graph = [(i1, i2) for i1 in range(num_images) for i2 in range(i1 + 1, num_images)]
    with (
        LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=False, dashboard_address=":0") as cluster,
        Client(cluster) as client,
    ):
        image_future_map = dict(enumerate(client.scatter(images)))
        keypoints_list, v_corr_idxs_dict = verify_visibility_graph(
            client,
            DetDescCorrespondenceGenerator(matcher=_StubMatcher(), detector_descriptor=_StubDetectorDescriptor()),
            _StubTwoViewEstimator(),
            _StubLoader(num_images),
            image_future_map,
            {i: _StubOneViewData() for i in range(num_images)},
            visibility_graph,
        )
    return keypoints_list, v_corr_idxs_dict, visibility_graph


class TestVerifyVisibilityGraph(unittest.TestCase):
    def test_inline_and_parallel_arms_agree(self) -> None:
        """The 1-worker (inline) and multi-worker (fan-out) arms must produce identical results."""
        keypoints_inline, v_corr_inline, _ = _run_with_workers(n_workers=1)
        keypoints_parallel, v_corr_parallel, _ = _run_with_workers(n_workers=2)

        self.assertEqual(len(keypoints_inline), len(keypoints_parallel))
        for kp_a, kp_b in zip(keypoints_inline, keypoints_parallel):
            np.testing.assert_array_equal(kp_a.coordinates, kp_b.coordinates)
        self.assertEqual(set(v_corr_inline.keys()), set(v_corr_parallel.keys()))
        for edge in v_corr_inline:
            np.testing.assert_array_equal(v_corr_inline[edge], v_corr_parallel[edge])

    def test_only_verified_edges_survive(self) -> None:
        """The verified graph keeps exactly the edges the two-view estimator validates."""
        keypoints_list, v_corr_idxs_dict, visibility_graph = _run_with_workers(n_workers=1, num_images=6)
        expected_edges = {(i1, i2) for (i1, i2) in visibility_graph if (i1 + i2) % 2 == 0}
        self.assertEqual(set(v_corr_idxs_dict.keys()), expected_edges)
        self.assertLess(len(v_corr_idxs_dict), len(visibility_graph))
        self.assertEqual(len(keypoints_list), 6)


if __name__ == "__main__":
    unittest.main()
