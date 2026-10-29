"""SSH history stays redacted while live retries retain their private identity path."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from visualization import job_history, runtime


def _wait_for_jobs(manager: runtime.JobManager) -> None:
    """Wait for mocked SSH lifecycle threads without launching a remote process."""
    for thread in manager._threads:
        thread.join(timeout=3)
        assert not thread.is_alive()


def _ssh_spec(identity: Path) -> dict:
    """Build a key-auth request containing distinct values for redaction checks."""
    return {
        "name": "SSH history",
        "loader": "olsson",
        "execution_target": "remote",
        "remote_connection": "ssh",
        "ssh_host": "cyber",
        "ssh_workspace": "/remote/gtsfm",
        "ssh_authentication": "key",
        "ssh_private_key": str(identity),
        "api_key": "api-secret-must-not-persist",
        "modal_token_secret": "modal-secret-must-not-persist",
    }


@pytest.fixture
def ssh_history(tmp_path, monkeypatch):
    """Use real history files and job lifecycle with a mocked SSH transport."""
    identity = tmp_path / "private-identity-file"
    identity.write_text("test fixture, not a real private key")
    received, written = [], []

    class Workspace:
        endpoint = "http://127.0.0.1:12345"
        api_key = "temporary-service-secret"
        stderr_tail = ""

        def __init__(self, spec):
            received.append(deepcopy(spec))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            pass

        def ready(self, read_json):
            return {"devices": []}

    monkeypatch.setattr(runtime, "SSHWorkspace", Workspace)
    write_record = job_history.write_record

    def capture_record(directory, record):
        written.append(deepcopy(record))
        write_record(directory, record)

    monkeypatch.setattr(job_history, "write_record", capture_record)
    manager = runtime.JobManager(tmp_path / "results")
    yield manager, identity, received, written
    manager.shutdown()


@pytest.mark.parametrize("status", ["failed", "completed"])
def test_ssh_terminal_history_excludes_private_state_and_credentials(ssh_history, monkeypatch, status):
    """Both successful and failed SSH lifecycles persist their redacted terminal state."""
    manager, identity, received, written = ssh_history

    def finish(job):
        job.status = status
        job.return_code = 0 if status == "completed" else 1

    monkeypatch.setattr(manager, "_run_remote", finish)
    created = manager.start(_ssh_spec(identity))
    _wait_for_jobs(manager)
    job = manager.get(created["id"])
    assert received[0]["ssh_private_key"] == str(identity)
    assert job.ssh_spec is None
    assert job.ssh_identity_file == str(identity)
    assert not job_history.SECRET_FIELDS.intersection(job.spec)
    assert not job_history.SECRET_FIELDS.intersection(job.public()["spec"])

    record = json.loads((manager.history_root / f"{job.id}.json").read_text())["job"]
    assert record["status"] == status
    assert record["return_code"] == job.return_code
    assert len(written) >= 2  # Queued state and the terminal state from _run_ssh's finally block.
    for saved in written:
        assert "ssh_spec" not in saved
        assert "ssh_identity_file" not in saved
        assert "remote_api_key" not in saved
        serialized = json.dumps(saved)
        assert str(identity) not in serialized
        assert "api-secret-must-not-persist" not in serialized
        assert "modal-secret-must-not-persist" not in serialized
        assert "temporary-service-secret" not in serialized


def test_live_key_auth_retry_reconstructs_identity_from_volatile_state(ssh_history, monkeypatch):
    """Retry uses the private identity field after the original ssh_spec has been discarded."""
    manager, identity, received, _ = ssh_history
    monkeypatch.setattr(manager, "_run_remote", lambda job: setattr(job, "status", "failed"))
    original = manager.start(_ssh_spec(identity))
    _wait_for_jobs(manager)
    assert manager.get(original["id"]).ssh_spec is None
    assert "ssh_private_key" not in manager.get(original["id"]).spec

    retried = manager.retry(original["id"])
    _wait_for_jobs(manager)
    assert retried["id"] != original["id"]
    assert len(received) == 2
    assert received[1]["ssh_authentication"] == "key"
    assert received[1]["ssh_private_key"] == str(identity)
    assert manager.get(retried["id"]).retry_of == original["id"]
    assert manager.get(retried["id"]).ssh_identity_file == str(identity)
    assert not job_history.SECRET_FIELDS.intersection(retried["spec"])


def test_retry_waits_for_the_original_ssh_cleanup(ssh_history, monkeypatch):
    """A failed state cannot start another GPU run while its old transport is still owned."""
    manager, identity, _, _ = ssh_history
    monkeypatch.setattr(manager, "_run_remote", lambda job: setattr(job, "status", "failed"))
    original = manager.start(_ssh_spec(identity))
    _wait_for_jobs(manager)
    manager._ssh_workspaces[original["id"]] = object()
    try:
        with pytest.raises(ValueError, match="SSH cleanup is still in progress"):
            manager.retry(original["id"])
    finally:
        manager._ssh_workspaces.pop(original["id"])


@pytest.mark.parametrize("saved_status", ["failed", "running"])
def test_restored_ssh_history_never_rehydrates_private_state_or_allows_retry(ssh_history, monkeypatch, saved_status):
    """Older records cannot restore credentials or make a remote retry appear usable."""
    manager, identity, _, _ = ssh_history
    monkeypatch.setattr(manager, "_run_remote", lambda job: setattr(job, "status", "failed"))
    created = manager.start(_ssh_spec(identity))
    _wait_for_jobs(manager)
    path = manager.history_root / f"{created['id']}.json"
    payload = json.loads(path.read_text())
    payload["job"]["status"] = saved_status
    # Simulate a record from an older buggy writer, bypassing today's redacting writer.
    payload["job"].update(
        ssh_spec=_ssh_spec(identity), ssh_identity_file=str(identity), remote_api_key="stale-service-secret"
    )
    payload["job"]["spec"].update(ssh_private_key=str(identity), api_key="stale-api-secret")
    path.write_text(json.dumps(payload))

    restored = runtime.JobManager(manager.results_root)
    job = restored.get(created["id"])
    assert job.status == ("interrupted" if saved_status == "running" else "failed")
    assert job.ssh_spec is None
    assert job.ssh_identity_file is None
    assert job.remote_api_key is None
    assert not job_history.SECRET_FIELDS.intersection(job.spec)
    assert not job.retry_available
    assert not job.public()["retry_available"]
    with pytest.raises(ValueError, match="Re-enter remote credentials"):
        restored.retry(job.id)
    restored._save_job(job)
    sanitized = path.read_text()
    assert "ssh_spec" not in sanitized
    assert "ssh_identity_file" not in sanitized
    assert str(identity) not in sanitized
    assert "stale-service-secret" not in sanitized
    assert "stale-api-secret" not in sanitized


def test_failed_ssh_startup_is_saved_by_terminal_finally(ssh_history, monkeypatch):
    """A startup failure before any remote job exists still records its final outcome."""
    manager, identity, received, _ = ssh_history

    def fail_readiness(self, read_json):
        raise ValueError("remote Python environment is unavailable")

    monkeypatch.setattr(runtime.SSHWorkspace, "ready", fail_readiness)
    created = manager.start(_ssh_spec(identity))
    _wait_for_jobs(manager)
    assert len(received) == 3
    record = json.loads((manager.history_root / f"{created['id']}.json").read_text())["job"]
    assert record["status"] == "failed"
    assert "remote Python environment is unavailable" in record["error"]
    assert "ssh_spec" not in record and "ssh_identity_file" not in record
