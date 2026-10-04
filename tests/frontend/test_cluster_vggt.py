"""Unit tests for VGGT cluster optimizer helpers."""

import unittest
from pathlib import Path
from unittest import mock

import torch

from gtsfm.cluster_optimizer.cluster_vggt import _VGGT_MODEL_CACHE, ClusterVGGT, _run_vggt_pipeline
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.frontend.geometry_transformer import GeometryTransformerOutput
from gtsfm.frontend.multi_view_tracker import MultiViewTracker, TrackingConfig
from gtsfm.frontend.vggt_geometry_transformer import VggtGeometryTransformer


def _make_dummy_geo_output(images: torch.Tensor) -> GeometryTransformerOutput:
    """Build a minimal GeometryTransformerOutput for testing."""
    n, _, h, w = images.shape
    return GeometryTransformerOutput(
        device=torch.device("cpu"),
        dtype=torch.float32,
        images=images,
        extrinsic=torch.eye(4).unsqueeze(0).expand(n, -1, -1)[:, :3, :],
        intrinsic=torch.eye(3).unsqueeze(0).expand(n, -1, -1),
        depth_map=torch.ones(n, h, w),
        depth_confidence=torch.ones(n, h, w),
        dense_points=torch.zeros(n, h, w, 3),
    )


class ClusterVGGTCacheTest(unittest.TestCase):
    """Ensure VGGT model caching behaves as expected."""

    def setUp(self) -> None:
        _VGGT_MODEL_CACHE.clear()

    @mock.patch("gtsfm.cluster_optimizer.cluster_vggt.load_model")
    def test_model_cached_across_invocations(self, mocked_load_model) -> None:
        """Repeated pipeline calls with a cache key should reuse the same model."""

        cached_model = mock.Mock(name="VGGTModel")
        mocked_load_model.return_value = cached_model

        transformer = VggtGeometryTransformer()
        tracker = MultiViewTracker(TrackingConfig(tracking=False))

        seen_models: list[object] = []

        def fake_predict(self, images, *, model=None, **kwargs):
            seen_models.append(model)
            return _make_dummy_geo_output(images)

        image_batch = torch.zeros((2, 3, 16, 16))
        original_coords = torch.zeros((2, 6))
        common_kwargs = dict(
            original_coords=original_coords,
            transformer=transformer,
            tracker=tracker,
            image_indices=(0, 1),
            image_names=None,
            weights_path=None,
        )

        with (
            mock.patch.object(VggtGeometryTransformer, "predict", fake_predict),
            mock.patch.object(MultiViewTracker, "build_gtsfm_data", return_value=GtsfmData(2)),
        ):
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

        optimizer = ClusterVGGT(weights_path="/tmp/vggt.pt", use_model_cache=False)
        self.assertEqual(optimizer._loader_kwargs["weights_path"], Path("/tmp/vggt.pt"))
        self.assertIsNone(optimizer._model_cache_key)


if __name__ == "__main__":
    unittest.main()
