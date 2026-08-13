"""Tests for live Gaussian optimization previews."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gtsfm.splat.live import publish_training_update


def test_live_preview_is_bounded_and_reports_full_splat_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_export_splats(**kwargs: object) -> None:
        captured.update(kwargs)
        Path(str(kwargs["save_to"])).write_bytes(b"preview")

    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(export_splats=fake_export_splats))
    monkeypatch.setenv("GTSFM_LIVE_DIR", str(tmp_path))
    monkeypatch.setenv("GTSFM_LIVE_PREVIEW_INTERVAL", "25")
    monkeypatch.setenv("GTSFM_LIVE_PREVIEW_MAX_SPLATS", "10")
    splats = {
        "means": torch.arange(300, dtype=torch.float32).reshape(100, 3),
        "scales": torch.zeros((100, 3)),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(100, 1),
        "opacities": torch.zeros((100, 1)),
        "sh0": torch.zeros((100, 1, 3)),
        "shN": torch.zeros((100, 15, 3)),
    }

    publish_training_update(splats, step=0, max_steps=100, loss=0.25)

    assert Path(str(captured["save_to"])).name == "live_splats.tmp.ply"
    assert captured["means"].shape == (10, 3)  # type: ignore[union-attr]
    assert (tmp_path / "live_splats.ply").read_bytes() == b"preview"
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["step"] == 1
    assert status["splat_count"] == 100
    assert status["preview_splat_count"] == 10
    assert status["preview_written"] is True

