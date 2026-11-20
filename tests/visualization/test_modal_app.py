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


def test_modal_source_mount_excludes_mutable_netvlad_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep runtime weight downloads out of the source snapshot sent to Modal."""
    from modal.file_pattern_matcher import FilePatternMatcher

    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    sys.modules.pop("visualization.modal_app", None)
    modal_app = importlib.import_module("visualization.modal_app")
    ignored = FilePatternMatcher(*modal_app.THIRDPARTY_EXCLUDES)
    source = tmp_path / "hloc" / "netvlad.py"
    weights = tmp_path / "hloc" / "weights" / "VGG16-NetVLAD-Pitts30K.mat"
    weights.parent.mkdir(parents=True)
    source.write_text("# Model implementation\n")
    weights.write_bytes(b"partially downloaded checkpoint")

    assert not ignored(source)
    assert ignored(weights)
    weights.write_bytes(b"changed during packaging")
    assert ignored(weights)


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
    assert first.status == "running"
    assert not result.exists()
    assert "Result sync is retrying" in first.log_tail[-1]

    assert manager.get("completed").status == "completed"
    assert result.read_text(encoding="utf-8") == "1 0 0 0 255 0 0\n"
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
    assert manager.list() == []


def test_gpu_terminal_state_is_published_only_after_volume_commit(monkeypatch, tmp_path):
    """A polling control service cannot download an uncommitted terminal result."""
    from types import SimpleNamespace

    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    for name in ("HF_HOME", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME"):
        monkeypatch.setenv(name, str(tmp_path))
    sys.modules.pop("visualization.modal_app", None)
    modal_app = importlib.import_module("visualization.modal_app")
    events = []
    payload = {"id": "finished", "status": "completed", "live_root": str(tmp_path / "live")}
    job = SimpleNamespace(status="completed", live_root=payload["live_root"], public=lambda: dict(payload))

    class State(dict):
        def __setitem__(self, key, value):
            events.append(("publish", value["status"]))
            super().__setitem__(key, value)

    class Volume:
        def commit(self):
            events.append(("commit", None))

    class Manager:
        def start(self, spec, *, job_id):
            assert spec["hardware"] == "cuda:0"

        def get(self, job_id):
            return job

    monkeypatch.setattr(modal_app, "workspace", Volume())
    monkeypatch.setattr(modal_app, "job_state", State())
    monkeypatch.setattr(modal_app, "WORKSPACE_MOUNT", str(tmp_path))
    monkeypatch.setattr(modal_app, "JobManager", lambda *args, **kwargs: Manager())

    assert modal_app.gtsfm_gpu_job.local("finished", {})["status"] == "completed"
    assert events == [("commit", None), ("publish", "completed")]


def test_result_download_waits_for_successful_volume_reload(monkeypatch, tmp_path):
    """Persistent volume reload failure must not produce an empty success archive."""
    from fastapi.testclient import TestClient

    from visualization.app import create_app

    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    sys.modules.pop("visualization.modal_app", None)
    modal_app = importlib.import_module("visualization.modal_app")
    payload = {"id": "finished", "status": "completed", "output_root": str(tmp_path / "runs" / "finished"), "spec": {}}

    class Volume:
        def reload(self):
            raise RuntimeError("Volume is temporarily unavailable")

    monkeypatch.setattr(modal_app, "workspace", Volume())
    monkeypatch.setattr(modal_app, "job_state", {"__job_ids__": ["finished"], "finished": payload})
    manager = modal_app.ModalGpuJobManager(tmp_path)
    with TestClient(create_app(tmp_path, job_manager=manager)) as client:
        response = client.get("/api/jobs/finished/results")
        assert response.status_code == 409
        assert "after the run completes" in response.json()["error"]


def test_current_deployment_is_the_only_reusable_configuration(monkeypatch, tmp_path):
    """A superseded deployment cannot provide the hardware behind the current endpoint."""
    import threading

    from visualization import modal_deployment

    monkeypatch.setattr(modal_deployment, "find_source_root", lambda: tmp_path)
    monkeypatch.setattr(modal_deployment, "source_identity", lambda _: {"fingerprint": "current"})
    manager = modal_deployment.ModalDeploymentManager(lambda *_: {})
    scheduled = []

    class Thread:
        def __init__(self, *, target, args, **kwargs):
            scheduled.append(args[0])

        def start(self):
            pass

    monkeypatch.setattr(threading, "Thread", Thread)
    first = manager.start("ak-test", "as-test", "L4")
    assert manager.start("ak-test", "as-test", "L4")["id"] == first["id"]
    with pytest.raises(ValueError, match="workspace setup is active"):
        manager.start("ak-test", "as-test", "L40S")
    scheduled[-1].status = "completed"
    second = manager.start("ak-test", "as-test", "L40S")
    scheduled[-1].status = "completed"

    assert second["id"] != first["id"]
    third = manager.start("ak-test", "as-test", "L4")
    assert third["id"] != first["id"]
    assert len(scheduled) == 3


def test_eager_gpu_status_and_native_cancellation_survive_controller_restart(monkeypatch, tmp_path):
    """The submit reply cannot overwrite early GPU progress or lose its call handle."""
    from types import SimpleNamespace

    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    sys.modules.pop("visualization.modal_app", None)
    modal_app = importlib.import_module("visualization.modal_app")
    state = {}
    calls = []

    def spawn(job_id, spec):
        assert "api_key" not in spec
        state[job_id] = {**state[job_id], "status": "running", "log_tail": ["gpu began"]}
        return SimpleNamespace(object_id="fc-test")

    class Call:
        def get(self, *, timeout):
            raise TimeoutError

        def cancel(self, *, terminate_containers):
            assert terminate_containers
            calls.append("cancelled")

    def from_id(call_id):
        assert call_id == "fc-test"
        return Call()

    monkeypatch.setattr(modal_app, "job_state", state)
    monkeypatch.setattr(modal_app, "workspace", SimpleNamespace(commit=lambda: None, reload=lambda: None))
    monkeypatch.setattr(modal_app, "gtsfm_gpu_job", SimpleNamespace(spawn=spawn))
    monkeypatch.setattr(modal_app.modal, "FunctionCall", SimpleNamespace(from_id=from_id))

    manager = modal_app.ModalGpuJobManager(tmp_path)
    created = manager.start({"hardware": "cuda:0", "api_key": "should-stay-local"})
    restored = modal_app.ModalGpuJobManager(tmp_path)
    assert restored.get(created["id"]).public()["log_tail"] == ["gpu began"]
    with pytest.raises(ValueError, match="already running"):
        restored.start({"hardware": "cuda:0"})
    assert restored.cancel(created["id"])["status"] == "cancelled"
    assert calls == ["cancelled"]
