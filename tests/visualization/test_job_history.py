"""Run history survives server restarts without silently relaunching pipelines."""

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visualization import runtime
from visualization.app import create_app

_RUN_PROCESS = runtime.JobManager._run


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.JobManager, "_run", lambda *_args: None)
    monkeypatch.setattr(runtime.JobManager, "_run_remote", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "detect_hardware",
        lambda: {"devices": [{"id": "cpu", "status": "available", "supports_gaussian_splatting": False}]},
    )
    return runtime.JobManager(tmp_path)


def start(manager, **overrides):
    payload = manager.start(
        {
            "name": "Saved run",
            "dataset_dir": str(manager.results_root),
            "loader": "olsson",
            "config_name": "megaloc_sift_metis",
            "hardware": "cpu",
            "splat_implementation": "none",
            "num_workers": 1,
            "threads_per_worker": 16,
            "worker_memory_limit": "23GB",
            **overrides,
        }
    )
    return manager.get(payload["id"])


@pytest.mark.parametrize("exit_code", [0, 3])
def test_process_outcome_settings_and_logs_survive_restart(manager, exit_code):
    job = start(manager)
    # Exercise the real process/log lifecycle with a tiny command, never reconstruction.
    job.command = [sys.executable, "-c", f"print('saved pipeline output'); raise SystemExit({exit_code})"]
    _RUN_PROCESS(manager, job, os.environ.copy())
    before = job.public()
    restored = runtime.JobManager(manager.results_root)
    assert restored.get(job.id).public() == before
    assert restored.get(job.id).process is None
    assert restored.get(job.id).return_code == exit_code
    client = TestClient(create_app(manager.results_root, job_manager=restored))
    assert client.get("/api/jobs").json()["items"][0] == before
    assert "saved pipeline output" in restored.diagnostic_prompt(job.id)
    if exit_code:
        retried = restored.retry(job.id)
        assert retried["spec"] == before["spec"]
        assert retried["output_root"] != before["output_root"]
        assert "gtsfm.runner" in retried["command"]
        assert runtime.JobManager(manager.results_root).get(retried["id"]).retry_of == job.id


def test_cancelled_and_interrupted_runs_are_not_relaunched(manager):
    cancelled = start(manager)
    manager.cancel(cancelled.id)
    interrupted = start(manager)
    interrupted.pid = os.getpid()
    interrupted.status = "running"
    manager._save_job(interrupted)
    restored = runtime.JobManager(manager.results_root)
    assert restored.get(cancelled.id).status == "cancelled"
    job = restored.get(interrupted.id)
    assert job.status == "interrupted"
    assert job.process is None
    assert "not automatically restarted" in job.history_note
    with pytest.raises(ValueError, match="still exists"):
        restored.retry(job.id)
    # Do not launch a duplicate until the previous process is known to be gone.
    job.pid = None
    retry = restored.retry(job.id)
    assert retry["status"] == "queued"
    child = restored.get(retry["id"])
    child.pid = os.getpid()
    restored._save_job(job)
    restored._save_job(child)
    again = runtime.JobManager(manager.results_root)
    with pytest.raises(ValueError, match="still exists"):
        again.retry(job.id)


def test_history_is_newest_first_ignores_corruption_and_refreshes_log_tail(manager):
    oldest = start(manager, name="A")
    newest = start(manager, name="Z")
    for job, created in [(oldest, "2026-10-24T23:00:00+00:00"), (newest, "2026-10-25T01:00:00+00:00")]:
        job.created_at = created
        job.status = "failed"
        manager._save_job(job)
    (manager.runtime_root / f"{newest.id}.log").write_text("\n".join(f"line {i}" for i in range(300)))
    (manager.history_root / "broken.json").write_text("{")
    (manager.history_root / "unfinished.tmp").write_text("{")
    restored = runtime.JobManager(manager.results_root)
    assert [job["id"] for job in restored.list()] == [newest.id, oldest.id]
    assert restored.get(newest.id).log_tail == [f"line {i}" for i in range(50, 300)]
    assert (manager.history_root / "broken.json").read_text() == "{"


def test_atomic_history_write_leaves_previous_record_on_failure(manager, monkeypatch):
    job = start(manager)
    path = manager.history_root / f"{job.id}.json"
    original = path.read_bytes()
    assert path.stat().st_mode & 0o777 == 0o600

    def fail_replace(*_args):
        raise OSError("Disk unavailable")

    monkeypatch.setattr(Path, "replace", fail_replace)
    job.status = "failed"
    with pytest.raises(OSError, match="Disk unavailable"):
        manager._save_job(job, required=True)
    assert path.read_bytes() == original
    assert list(manager.history_root.glob("*.tmp")) == []


def test_remote_credentials_are_never_saved_or_exposed(manager):
    job = start(manager, execution_target="remote", remote_endpoint="https://example.invalid", api_key="private-secret")
    job.status = "failed"
    manager._save_job(job)
    assert "private-secret" not in (manager.history_root / f"{job.id}.json").read_text()
    restored = runtime.JobManager(manager.results_root)
    assert restored.get(job.id).status == "failed"
    assert restored.get(job.id).remote_api_key is None
    assert not restored.get(job.id).retry_available
    assert "private-secret" not in json.dumps(restored.list())


def test_legacy_logs_recover_history_without_inventing_settings(tmp_path):
    root = tmp_path / ".gtsfm"
    root.mkdir()
    logs = {
        "000000000001": "2026-10-24 10:00:00 INFO: Config File: megaloc_sift_metis\nShutting down Dask client...\n",
        "000000000002": (
            "2026-10-24 11:00:00 INFO: Config File: vggt\nTraceback (most recent call last):\nRuntimeError: OOM\n"
        ),
        "000000000003": "2026-10-24 12:00:00 INFO: matching pairs...\n",
    }
    for job_id, text in logs.items():
        (root / f"{job_id}.log").write_text(text)
    output = tmp_path / "runs" / "Ravenna-000000000001"
    output.mkdir(parents=True)
    manager = runtime.JobManager(tmp_path)
    assert [job["id"] for job in manager.list()] == list(reversed(logs))
    assert [job["status"] for job in manager.list()] == ["interrupted", "failed", "completed"]
    recovered = manager.get("000000000001")
    assert recovered.name == "Ravenna"
    assert recovered.output_root == str(output)
    assert recovered.spec["config_name"] == "megaloc_sift_metis"
    assert recovered.created_at == "2026-10-24T10:00:00+00:00"
    assert recovered.command == [] and recovered.return_code is None
    assert not recovered.retry_available
    assert "original command" in recovered.history_note
    with pytest.raises(ValueError, match="Retry is unavailable"):
        manager.retry("000000000002")
    assert len(runtime.JobManager(tmp_path).list()) == 3
