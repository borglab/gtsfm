"""The tracker must not retain backbone activations during correlation allocation."""

import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from gtsfm.frontend.multi_view_tracker import _predict_vggt_tracks


@pytest.mark.parametrize("free_bytes, expected_flushes", [(None, 0), (0, 1), (40 * 1024**3, 0)])
def test_backbone_tokens_are_released_before_tracking(monkeypatch, free_bytes, expected_flushes):
    token_refs = []
    images = SimpleNamespace(is_cuda=free_bytes is not None, device=torch.device("cuda"))
    empty_cache = Mock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_bytes, 48 * 1024**3))
    query_points = torch.rand(1, 2048, 2)

    def aggregate(received_images):
        assert received_images is images
        tokens = [torch.ones(1, 4, 128, 8, 8) for _ in range(4)]
        token_refs.extend(weakref.ref(tensor) for tensor in tokens)
        return tokens, 5

    def extract(tokens, received_images, patch_start):
        assert received_images is images
        assert patch_start == 5
        assert all(ref() is tensor for ref, tensor in zip(token_refs, tokens))
        return torch.stack(tokens).mean(dim=0)

    def track(*, query_points, fmaps, iters):
        assert all(ref() is None for ref in token_refs), "Backbone tensors still occupy memory during tracking"
        # Keep all 2048 queries in the same call: splitting would change spatial attention.
        assert query_points.shape == (1, 2048, 2)
        assert iters == 4
        torch.testing.assert_close(fmaps, torch.ones(1, 4, 128, 8, 8))
        return [query_points[:, None].expand(-1, 4, -1, -1)], torch.ones(1, 4, 2048), torch.ones(1, 4, 2048)

    model = SimpleNamespace(
        aggregator=aggregate,
        track_head=SimpleNamespace(feature_extractor=extract, tracker=track, iters=4),
    )
    with torch.no_grad():
        tracks, vis, conf = _predict_vggt_tracks(model, images, query_points)
    torch.testing.assert_close(tracks[-1], query_points[:, None].expand(-1, 4, -1, -1))
    assert vis.shape == conf.shape == (1, 4, 2048)
    assert empty_cache.call_count == expected_flushes
