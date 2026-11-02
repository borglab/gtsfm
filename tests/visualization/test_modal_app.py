"""Tests for the Modal workspace import path."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def test_modal_app_import_in_container_does_not_need_a_local_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    monkeypatch.delenv("GTSFM_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("GTSFM_REMOTE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("visualization.modal_app", None)

    modal_app = importlib.import_module("visualization.modal_app")

    assert modal_app.app.name == "gtsfm-studio"
    assert modal_app.gtsfm_workspace is not None


def test_modal_app_import_locally_requires_a_source_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MODAL_IS_REMOTE", raising=False)
    monkeypatch.delenv("GTSFM_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("GTSFM_REMOTE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("visualization.modal_app", None)

    with pytest.raises(RuntimeError, match="GTSFM_SOURCE_ROOT must point to a GTSFM source checkout"):
        importlib.import_module("visualization.modal_app")


class _FakeRequest:
    def __init__(self, method: str, path: str) -> None:
        self.method = method
        self.url = type("URL", (), {"path": path})()


def test_modal_workspace_allows_unauthenticated_catalog_and_scene_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    sys.modules.pop("visualization.modal_app", None)
    from visualization.modal_app import _is_public_workspace_read

    assert _is_public_workspace_read(_FakeRequest("GET", "/api/configuration"))
    assert _is_public_workspace_read(_FakeRequest("GET", "/api/samples"))
    assert _is_public_workspace_read(_FakeRequest("GET", "/api/scenes"))
    assert _is_public_workspace_read(_FakeRequest("GET", "/data/run/points3D.txt"))
    assert not _is_public_workspace_read(_FakeRequest("POST", "/api/jobs"))
    assert not _is_public_workspace_read(_FakeRequest("GET", "/api/jobs"))


def test_terminal_volume_reload_retries_until_results_are_visible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Retry a failed terminal sync before treating the result volume as current."""
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    sys.modules.pop("visualization.modal_app", None)
    modal_app = importlib.import_module("visualization.modal_app")
    output = tmp_path / "runs" / "completed"
    result = output / "results" / "points3D.txt"
    payload = {
        "id": "completed",
        "name": "completed",
        "status": "completed",
        "output_root": str(output),
        "spec": {},
    }

    class ResultVolume:
        def __init__(self) -> None:
            self.reloads = 0

        def reload(self) -> None:
            self.reloads += 1
            if self.reloads == 1:
                raise RuntimeError("Volume is temporarily unavailable")
            result.parent.mkdir(parents=True)
            result.write_text("1 0 0 0 255 0 0\n", encoding="utf-8")

    volume = ResultVolume()
    monkeypatch.setattr(modal_app, "workspace", volume)
    monkeypatch.setattr(modal_app, "job_state", {"__job_ids__": ["completed"], "completed": payload})
    manager = modal_app.ModalGpuJobManager(tmp_path)

    first = manager.get("completed")
    assert first.status == "completed"
    assert not result.exists()
    assert "Result sync is retrying" in first.log_tail[-1]
    assert "completed" not in manager._terminal_reloads

    assert manager.get("completed").status == "completed"
    assert result.read_text(encoding="utf-8") == "1 0 0 0 255 0 0\n"
    assert "completed" in manager._terminal_reloads
    manager.list()
    assert volume.reloads == 2


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "rocm:0"])
def test_modal_rejects_device_choices_it_cannot_honor_before_spawning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str
) -> None:
    """Reject unsupported devices before committing inputs or allocating a GPU."""
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    sys.modules.pop("visualization.modal_app", None)
    modal_app = importlib.import_module("visualization.modal_app")
    monkeypatch.setattr(modal_app, "job_state", {})
    manager = modal_app.ModalGpuJobManager(tmp_path)

    with pytest.raises(ValueError, match="runs on cuda:0"):
        manager.start({"execution_target": "local", "hardware": device})
    assert manager._jobs == {}
