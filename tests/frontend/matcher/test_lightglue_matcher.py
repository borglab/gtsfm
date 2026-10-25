"""Regression tests for sharing a lazily loaded LightGlue matcher across threads."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

import cloudpickle
import numpy as np
import pytest
import torch

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.lightglue_matcher import LightGlueMatcher
from thirdparty.LightGlue.lightglue import lightglue


class _FakeLightGlue(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        # Release the GIL during loading, as real weight loading does.
        time.sleep(0.02)
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, data):
        assert data["image0"]["descriptors"].device == self.weight.device
        return {"matches": [torch.tensor([[0, 0]], device=self.weight.device)]}


def _match(matcher):
    keypoints = Keypoints(coordinates=np.ones((2, 2)), responses=np.ones(2))
    descriptors = np.ones((2, 256))
    return matcher.match(keypoints, keypoints, descriptors, descriptors, (20, 20, 3), (20, 20, 3))


def test_concurrent_first_matches_load_and_place_model_once(monkeypatch):
    """Concurrent cold-cache matches must never replace or move an in-use model."""
    matcher = LightGlueMatcher("superpoint", use_cuda=False)
    start = Barrier(8)
    inference = Barrier(8)
    placements = []

    class Model(_FakeLightGlue):
        def to(self, device):
            assert matcher._model is None, "Model published before device placement finished"
            placements.append(device)
            return super().to(device)

        def forward(self, data):
            # Initialization is serialized, but inference must remain concurrent.
            inference.wait(timeout=10)
            return super().forward(data)

    factory = Mock(side_effect=Model)
    monkeypatch.setattr(lightglue, "LightGlue", factory)

    def match_pair(_):
        start.wait(timeout=10)
        return _match(matcher)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(match_pair, range(8)))
    factory.assert_called_once_with(features="superpoint")
    assert placements == [torch.device("cpu")]
    for result in results:
        np.testing.assert_array_equal(result, [[0, 0]])


@pytest.mark.parametrize("loaded", [False, True])
def test_dask_serialization(loaded, monkeypatch):
    """Both lazy and initialized matchers survive serialization with a new local lock."""
    monkeypatch.setattr(lightglue, "LightGlue", _FakeLightGlue)
    matcher = LightGlueMatcher("superpoint", use_cuda=False)
    if loaded:
        _match(matcher)
    restored = cloudpickle.loads(cloudpickle.dumps(matcher))
    assert restored._model_lock is not matcher._model_lock
    assert (restored._model is not None) == loaded
    np.testing.assert_array_equal(_match(restored), [[0, 0]])


def test_failed_initialization_can_retry(monkeypatch):
    """A failed load must not leave a partial model or locked matcher behind."""
    monkeypatch.setattr(lightglue, "LightGlue", Mock(side_effect=RuntimeError("load failed")))
    matcher = LightGlueMatcher("superpoint", use_cuda=False)
    with pytest.raises(RuntimeError, match="load failed"):
        _match(matcher)
    assert matcher._model is None
    monkeypatch.setattr(lightglue, "LightGlue", _FakeLightGlue)
    np.testing.assert_array_equal(_match(matcher), [[0, 0]])
