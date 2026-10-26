"""Unit tests for VGGT cluster optimizer helpers."""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

import gtsfm.cluster_optimizer.cluster_vggt as cluster_vggt
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


@pytest.mark.parametrize("cache_key", ["threaded-model", None])
def test_threaded_pipelines_are_exclusive_through_tracking_and_offload(monkeypatch, cache_key):
    """A 16-thread worker must run one whole VGGT pipeline at a time, even without caching."""
    monkeypatch.setattr(cluster_vggt, "_VGGT_MODEL_CACHE", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    start = Barrier(16)
    events = []
    model = object()

    def load(**kwargs):
        time.sleep(0.01)  # Model loading releases the GIL; cold-cache calls used to overlap here.
        return model

    load_model = mock.Mock(side_effect=load)
    monkeypatch.setattr(cluster_vggt, "load_model", load_model)

    def predict(images, *, model, **kwargs):
        assert model is expected_model
        events.append("predict")
        time.sleep(0.01)
        output = _make_dummy_geo_output(images)
        # Exercise the GPU offload path without allocating real CUDA tensors.
        output.device = torch.device("cuda")
        return output

    expected_model = model if cache_key is not None else None
    transformer = SimpleNamespace(predict=predict, config=SimpleNamespace(confidence_threshold=5.0, max_num_points=100))

    def track(*args, **kwargs):
        events.append("track")
        time.sleep(0.01)

    def offload(model):
        assert model is expected_model
        events.append("offload")
        time.sleep(0.01)

    def points(*args, **kwargs):
        events.append("points")
        return np.empty((0, 3)), np.empty((0, 3))

    def build(*args, **kwargs):
        events.append("build")
        return GtsfmData(2)

    tracker = SimpleNamespace(config=SimpleNamespace(tracking=True), run_tracking=track, build_gtsfm_data=build)
    monkeypatch.setattr(cluster_vggt, "offload_vggt_model", offload)
    monkeypatch.setattr(cluster_vggt, "high_confidence_pointcloud", points)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("cleanup"))

    def run(i):
        start.wait(timeout=10)
        return _run_vggt_pipeline(
            torch.zeros(2, 3, 16, 16),
            torch.zeros(2, 6),
            transformer=transformer,
            tracker=tracker,
            image_indices=(0, 1),
            model_cache_key=cache_key,
            seed=i,
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(run, range(16)))
    assert len(results) == 16
    assert events == ["predict", "track", "offload", "points", "build", "cleanup"] * 16
    assert load_model.call_count == (0 if cache_key is None else 1)


def test_pipeline_failure_releases_worker_lock(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    transformer = mock.Mock()
    transformer.predict.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        _run_vggt_pipeline(
            torch.zeros(2, 3, 16, 16),
            torch.zeros(2, 6),
            transformer=transformer,
            tracker=mock.Mock(),
            image_indices=(0, 1),
        )
    assert cluster_vggt._VGGT_PIPELINE_LOCK.acquire(blocking=False)
    cluster_vggt._VGGT_PIPELINE_LOCK.release()


if __name__ == "__main__":
    unittest.main()
