"""Unit tests for VGGT cluster optimizer helpers."""

import unittest
from pathlib import Path
from unittest import mock

import torch

from gtsfm.cluster_optimizer.cluster_vggt import _VGGT_MODEL_CACHE, ClusterVGGT, _run_vggt_pipeline


class ClusterVGGTCacheTest(unittest.TestCase):
    """Ensure VGGT model caching behaves as expected."""

    def setUp(self) -> None:
        _VGGT_MODEL_CACHE.clear()

    @mock.patch("gtsfm.cluster_optimizer.cluster_vggt.vggt.run_reconstruction")
    @mock.patch("gtsfm.cluster_optimizer.cluster_vggt.vggt.load_model")
    def test_model_cached_across_invocations(self, mocked_load_model, mocked_run_recon) -> None:
        """Repeated pipeline calls with a cache key should reuse the same model."""

        cached_model = mock.Mock(name="VGGTModel")
        mocked_load_model.return_value = cached_model

        seen_models: list[object] = []

        def fake_run_recon(*args, **kwargs):
            seen_models.append(kwargs.get("model"))
            return mock.Mock()

        mocked_run_recon.side_effect = fake_run_recon

        image_batch = torch.zeros((2, 3, 16, 16))
        original_coords = torch.zeros((2, 6))
        common_kwargs = dict(
            original_coords=original_coords,
            image_indices=(0, 1),
            image_names=None,
            config=None,
            weights_path=None,
        )

        _run_vggt_pipeline(
            image_batch,
            seed=0,
            model_cache_key=("cache", None),
            loader_kwargs={},
            **common_kwargs,
        )
        _run_vggt_pipeline(
            image_batch,
            seed=1,
            model_cache_key=("cache", None),
            loader_kwargs={},
            **common_kwargs,
        )

        mocked_load_model.assert_called_once()
        self.assertEqual(len(seen_models), 2)
        self.assertIs(seen_models[0], cached_model)
        self.assertIs(seen_models[0], seen_models[1])

    def test_cluster_configures_loader_kwargs(self) -> None:
        """ClusterVGGT should derive loader kwargs from the provided weights path."""

        optimizer = ClusterVGGT(weights_path="/tmp/vggt.pt", model_cache_key=False)
        self.assertEqual(optimizer._loader_kwargs["weights_path"], Path("/tmp/vggt.pt"))
        self.assertIsNone(optimizer._model_cache_key)


if __name__ == "__main__":
    unittest.main()
