"""Regression tests for concurrent NetVLAD batches on a threaded Dask worker."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

import cloudpickle
import numpy as np
import pytest
import torch

import gtsfm.frontend.global_descriptor.netvlad_global_descriptor as netvlad_module


class _FakeNetVLAD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        time.sleep(0.02)
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, data):
        images = data["image"]
        assert images.device == self.weight.device
        return {"global_descriptor": images.mean(dim=(2, 3)) * self.weight}


def test_concurrent_batches_initialize_once(monkeypatch):
    descriptor = netvlad_module.NetVLADGlobalDescriptor()
    start, inference = Barrier(8), Barrier(8)
    placements = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class Model(_FakeNetVLAD):
        def to(self, device):
            assert descriptor._model is None, "Model published before device placement finished"
            placements.append(device)
            return super().to(device)

        def forward(self, data):
            inference.wait(timeout=10)
            return super().forward(data)

    factory = Mock(side_effect=Model)
    monkeypatch.setattr(netvlad_module, "NetVLAD", factory)

    def describe_batch(i):
        start.wait(timeout=10)
        return descriptor.describe_batch(torch.full((4, 3, 8, 8), float(i)))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(describe_batch, range(8)))
    factory.assert_called_once_with()
    assert placements == [torch.device("cpu")]
    for i, batch in enumerate(results):
        np.testing.assert_array_equal(batch, np.full((4, 3), i))


@pytest.mark.parametrize("loaded", [False, True])
def test_dask_serialization(loaded, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(netvlad_module, "NetVLAD", _FakeNetVLAD)
    descriptor = netvlad_module.NetVLADGlobalDescriptor()
    images = torch.ones((4, 3, 8, 8))
    if loaded:
        descriptor.describe_batch(images)
    restored = cloudpickle.loads(cloudpickle.dumps(descriptor))
    assert restored._model_lock is not descriptor._model_lock
    assert (restored._model is not None) == loaded
    np.testing.assert_array_equal(restored.describe_batch(images), np.ones((4, 3)))
