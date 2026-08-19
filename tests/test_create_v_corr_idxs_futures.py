"""Unit tests for the parallel global two-view frontend (``create_v_corr_idxs_futures``).

The parallel path chunks the pairs and runs ``create_v_corr_idxs_inline`` on each chunk across the Dask
worker pool, scattering the shared read-only inputs once. These tests assert it returns EXACTLY the same
``{(i1, i2): v_corr_idxs}`` dict as the serial inline reference for any chunking, and that the ``valid()``
filter is honored. A deterministic stub estimator isolates the orchestration (chunking / scatter-as-blob /
merge) from real two-view geometry; cross-process worker pickling is exercised separately by end-to-end runs.

Authors: Kathirvel Gounder
"""

import unittest

import numpy as np
from dask.distributed import Client, LocalCluster

from gtsfm.common.keypoints import Keypoints
from gtsfm.two_view_estimator import create_v_corr_idxs_futures, create_v_corr_idxs_inline


class _StubResult:
    """Minimal stand-in for TwoViewResult exposing only what the frontend reduction reads."""

    def __init__(self, v_corr_idxs: np.ndarray, is_valid: bool) -> None:
        self.v_corr_idxs = v_corr_idxs
        self._is_valid = is_valid

    def valid(self) -> bool:
        return self._is_valid


class _StubTwoViewEstimator:
    """Deterministic stand-in for ``TwoViewEstimator.run_2view``.

    ``v_corr_idxs`` is derived from the pair's putative indices plus a per-pair offset, so inline and
    parallel results can be compared exactly and a chunk can never be confused for another. Pairs whose
    index sum is odd are marked invalid, exercising the ``valid()`` filter (invalid pairs must be dropped).
    """

    def run_2view(self, *, putative_corr_idxs, i1, i2, **kwargs) -> _StubResult:
        is_valid = (i1 + i2) % 2 == 0
        v_corr_idxs = putative_corr_idxs + (i1 * 1000 + i2)
        return _StubResult(v_corr_idxs, is_valid)


class _StubOneViewData:
    """Per-image data with the attributes the frontend reads (all unused by the stub estimator)."""

    def __init__(self) -> None:
        self.intrinsics = None
        self.camera_gt = None


def _make_inputs(num_images: int = 8):
    """Build deterministic keypoints, a fully-connected putative graph, priors, and per-view data."""
    keypoints_list = [Keypoints(np.full((5, 2), i, dtype=np.float32)) for i in range(num_images)]
    putative_corr_idxs_dict = {
        (i, j): np.arange(6, dtype=np.int32).reshape(3, 2) + (i + j)
        for i in range(num_images)
        for j in range(i + 1, num_images)
    }
    one_view_data_dict = {i: _StubOneViewData() for i in range(num_images)}
    relative_pose_priors: dict = {}
    return keypoints_list, putative_corr_idxs_dict, relative_pose_priors, one_view_data_dict


class TestCreateVCorrIdxsFutures(unittest.TestCase):
    """Parallel two-view must equal the serial inline reference for any chunking."""

    @classmethod
    def setUpClass(cls) -> None:
        # Threaded workers (processes=False): 2 workers, no cross-process pickling — deterministic and fast.
        # This still exercises chunking, scatter-as-blob, and the as_completed merge, which is the new code.
        cls.cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=False, dashboard_address=":0")
        cls.client = Client(cls.cluster)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        cls.cluster.close()

    def _assert_same(self, expected: dict, actual: dict) -> None:
        self.assertEqual(set(expected.keys()), set(actual.keys()))
        for edge in expected:
            np.testing.assert_array_equal(expected[edge], actual[edge])

    def test_parallel_matches_inline_default_chunking(self) -> None:
        """Default chunk sizing must reproduce the inline result exactly."""
        estimator = _StubTwoViewEstimator()
        keypoints_list, putative, priors, one_view = _make_inputs()

        expected = create_v_corr_idxs_inline(
            two_view_estimator=estimator,
            keypoints_list=keypoints_list,
            putative_corr_idxs_dict=putative,
            relative_pose_priors=priors,
            gt_scene_mesh=None,
            one_view_data_dict=one_view,
        )
        actual = create_v_corr_idxs_futures(
            self.client, estimator, keypoints_list, putative, priors, None, one_view
        )
        self._assert_same(expected, actual)
        # Sanity: the valid() filter dropped the odd-sum pairs (so it is not a trivial pass-through).
        self.assertTrue(all((i1 + i2) % 2 == 0 for (i1, i2) in actual))
        self.assertLess(len(actual), len(putative))

    def test_parallel_matches_inline_tiny_chunks(self) -> None:
        """Force many single-pair chunks (chunk_size=1) to stress the chunk/merge boundary."""
        estimator = _StubTwoViewEstimator()
        keypoints_list, putative, priors, one_view = _make_inputs()

        expected = create_v_corr_idxs_inline(
            two_view_estimator=estimator,
            keypoints_list=keypoints_list,
            putative_corr_idxs_dict=putative,
            relative_pose_priors=priors,
            gt_scene_mesh=None,
            one_view_data_dict=one_view,
        )
        actual = create_v_corr_idxs_futures(
            self.client, estimator, keypoints_list, putative, priors, None, one_view, chunk_size=1
        )
        self._assert_same(expected, actual)

    def test_single_chunk_matches_inline(self) -> None:
        """A chunk_size >= num_pairs (one chunk on one worker) must reproduce the inline result."""
        estimator = _StubTwoViewEstimator()
        keypoints_list, putative, priors, one_view = _make_inputs()

        expected = create_v_corr_idxs_inline(
            two_view_estimator=estimator,
            keypoints_list=keypoints_list,
            putative_corr_idxs_dict=putative,
            relative_pose_priors=priors,
            gt_scene_mesh=None,
            one_view_data_dict=one_view,
        )
        actual = create_v_corr_idxs_futures(
            self.client, estimator, keypoints_list, putative, priors, None, one_view, chunk_size=10_000
        )
        self._assert_same(expected, actual)

    def test_empty_graph_returns_empty(self) -> None:
        """No pairs → empty dict, no tasks submitted."""
        estimator = _StubTwoViewEstimator()
        keypoints_list, _, priors, one_view = _make_inputs()
        actual = create_v_corr_idxs_futures(self.client, estimator, keypoints_list, {}, priors, None, one_view)
        self.assertEqual(actual, {})


if __name__ == "__main__":
    unittest.main()
