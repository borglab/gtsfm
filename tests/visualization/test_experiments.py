"""Experiment identity, portable reuse, native account login and cancellation boundaries."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from visualization import app as app_runtime
from visualization import experiments, modal_account, runtime
from visualization.app import create_app
from visualization.modal_account import ModalAccount, ModalCredentials


def _job(root: Path, **settings: object) -> SimpleNamespace:
    """Build a minimal managed experiment for provenance tests."""
    spec = {"name": "door", "config_name": "vggt", **settings}
    job = SimpleNamespace(
        id="exp1",
        name="door",
        status="running",
        created_at="2026-11-13T00:00:00+00:00",
        output_root=str(root),
        spec=spec,
        command=["python", "-m", "gtsfm.runner"],
    )
    job.public = lambda: {"id": job.id, "name": job.name, "status": job.status}
    return job


def test_capture_records_input_contents_and_excludes_credentials(tmp_path: Path, monkeypatch) -> None:
    """A source/data record must change with input contents and never contain credentials."""
    dataset = tmp_path / "input"
    dataset.mkdir()
    (dataset / "one.jpg").write_bytes(b"first")
    monkeypatch.setattr(experiments, "source_identity", lambda: {"commit": "abc", "dirty": True, "fingerprint": "code"})
    job = _job(
        tmp_path / "output",
        dataset_dir=str(dataset),
        api_key="credential-marker-123",
        modal_token_secret="secret-marker-456",
    )
    experiments.capture_experiment(job)
    record = json.loads((Path(job.output_root) / "experiment.json").read_text())
    assert "credential-marker-123" not in json.dumps(record) and "secret-marker-456" not in json.dumps(record)
    assert record["source"]["dirty"] is True
    assert record["runtime"]["python"]
    assert record["dataset"]["files"][0]["path"] == "one.jpg"
    (dataset / "one.jpg").write_bytes(b"other")
    experiments.capture_experiment(job)
    changed = json.loads((Path(job.output_root) / "experiment.json").read_text())
    assert changed["dataset"]["manifest_sha256"] != record["dataset"]["manifest_sha256"]


def test_remote_experiment_preserves_submission_and_uses_execution_provenance(tmp_path: Path, monkeypatch) -> None:
    """The displayed experiment keeps its own identity and the actual execution source."""
    root = tmp_path / "output"
    job = _job(root, execution_target="remote", remote_endpoint="https://workspace.test")
    monkeypatch.setattr(experiments, "source_identity", lambda: {"commit": "submission"})
    submission = experiments.capture_experiment(job)
    assert submission is not None and submission["source"] is None

    remote = _job(root)
    remote.id = "provider-id"
    monkeypatch.setattr(experiments, "source_identity", lambda: {"commit": "execution"})
    experiments.capture_experiment(remote)
    assert experiments.experiment_details(job, [])["provenance"]["source"] is None

    experiments.record_remote_execution(root, submission, remote.id)
    record = experiments.experiment_details(job, [])["provenance"]
    assert record["id"] == job.id and record["execution_id"] == remote.id
    assert record["submission_source"]["commit"] == "submission"
    assert record["source"]["commit"] == "execution"
    assert record["settings"]["remote_endpoint"] == "https://workspace.test"


def test_sample_download_receipt_does_not_change_input_identity(tmp_path: Path) -> None:
    """A curated scene has the same input identity locally and after cloud download."""
    from visualization.datasets import dataset_fingerprint

    dataset = tmp_path / "input"
    dataset.mkdir()
    image = dataset / "one.jpg"
    image.write_bytes(b"same image bytes")
    job = _job(tmp_path / "output", dataset_dir=str(dataset))
    record = Path(job.output_root) / "experiment.json"
    experiments.capture_experiment(job)
    before = json.loads(record.read_text())["dataset"]
    fingerprint = dataset_fingerprint(dataset)
    (dataset / ".gtsfm-sample.json").write_text('{"id":"lund-door","downloaded":true}')
    experiments.capture_experiment(job)
    assert json.loads(record.read_text())["dataset"] == before
    assert dataset_fingerprint(dataset) == fingerprint
    image.write_bytes(b"different image bytes")
    experiments.capture_experiment(job)
    assert json.loads(record.read_text())["dataset"] != before
    assert dataset_fingerprint(dataset) != fingerprint


def test_portable_import_strips_machine_accounts_and_absolute_input_paths() -> None:
    """Importing a shared record cannot silently reuse another machine's input paths."""
    settings = experiments.validate_import(
        {
            "schema": experiments.SCHEMA,
            "settings": {
                "dataset_dir": "/private/photos",
                "images_dir": "/private/images",
                "remote_endpoint": "https://other",
                "ssh_host": "other",
                "modal_token_secret": "secret",
                "api_key": "secret",
                "hardware": "cuda:1",
                "execution_target": "remote",
                "config_name": "vggt",
                "sample_id": "lund-door",
                "loader_options": {"poses_fpath": "/private/poses", "mask_path": "masks", "maximum": 4},
                "hydra_overrides": {"model.foo": 2},
            },
        }
    )
    assert settings == {
        "config_name": "vggt",
        "sample_id": "lund-door",
        "loader_options": {"mask_path": "masks", "maximum": 4},
        "hydra_overrides": {"model.foo": 2},
    }
    with pytest.raises(ValueError):
        experiments.validate_import({"schema": "other", "settings": {}})


def test_details_preserve_measured_zero_and_filter_nonfinite_metrics(tmp_path: Path) -> None:
    """Metrics remain measured values; legacy records do not invent provenance."""
    root = tmp_path / "output"
    root.mkdir()
    (root / "metrics.json").write_text('{"optimization":{"error":0,"invalid":NaN,"count":4}}')
    (root / "config.yaml").write_text("model:\n  threshold: 0.5\n")
    (root / "experiment.json").write_text("[]")
    scenes = [
        {"rel_path": "run/vggt"},
        {"rel_path": "run/child/merged_retriangulated"},
        {"rel_path": "run/merged_retriangulated"},
    ]
    details = experiments.experiment_details(_job(root), scenes)
    assert details["metrics"] == {"optimization": {"error": 0, "count": 4}}
    assert details["configuration"] == {"model": {"threshold": 0.5}}
    assert details["provenance"]["source"] is None
    assert details["preferred_result"]["rel_path"] == "run/merged_retriangulated"


def test_experiment_api_exports_evidence_and_import_does_not_start_a_job(tmp_path: Path, monkeypatch) -> None:
    """Export and import use the same record schema, guarded by the workspace API key."""
    monkeypatch.setattr(runtime.JobManager, "_run_remote", lambda *_: None)
    app = create_app(tmp_path)
    client = TestClient(app)
    response = client.post(
        "/api/jobs",
        json={
            "loader": "olsson",
            "execution_target": "remote",
            "remote_endpoint": "https://workspace.test",
            "api_key": "hidden",
        },
    )
    assert response.status_code == 202
    job = app.state.gtsfm_jobs.get(response.json()["id"])
    experiments.capture_experiment(job)
    exported = client.get(f"/api/experiments/{job.id}/export")
    assert exported.status_code == 200
    assert "attachment" in exported.headers["content-disposition"]
    assert "hidden" not in exported.text
    assert client.post("/api/experiments/import", json=exported.json()).status_code == 200
    assert len(client.get("/api/jobs").json()["items"]) == 1
    assert client.get("/api/experiments/missing").status_code == 404
    monkeypatch.setenv("GTSFM_API_KEY", "required")
    assert client.get(f"/api/experiments/{job.id}/export").status_code == 401
    assert client.post("/api/experiments/import", json=exported.json()).status_code == 401
    app.state.gtsfm_jobs.cancel(job.id)


def test_native_profile_reuse_keeps_tokens_server_side(monkeypatch) -> None:
    """A successful native connection exposes workspace and profile names only."""
    credentials = ModalCredentials("research", "ak-private", "as-private")
    monkeypatch.setattr(modal_account, "profile_credentials", lambda *_: credentials)
    monkeypatch.setattr(modal_account, "native_profiles", lambda: ["research"])
    verify = AsyncMock(return_value="GTSfM team")
    monkeypatch.setattr(modal_account, "verify_profile", verify)
    account = ModalAccount()
    result = asyncio.run(account.connect("research"))
    assert result["status"] == "connected" and result["workspace"] == "GTSfM team"
    assert "ak-private" not in json.dumps(result) and "as-private" not in repr(credentials)
    verify.assert_awaited_once_with(credentials)


def test_native_profile_selection_never_changes_active_account(tmp_path: Path, monkeypatch) -> None:
    """Reading another native profile does not rewrite Modal's active selection."""
    import modal.config

    path = tmp_path / "modal.toml"
    path.write_text(
        '[first]\nactive=true\ntoken_id="ak-one"\ntoken_secret="as-one"\n'
        '[second]\ntoken_id="ak-two"\ntoken_secret="as-two"\n'
    )
    before = path.read_bytes()
    monkeypatch.setattr(modal.config, "user_config_path", str(path))
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    assert modal_account.profile_credentials().profile == "first"
    assert modal_account.profile_credentials("second").token_id == "ak-two"
    assert path.read_bytes() == before


def test_failed_native_verification_sanitizes_provider_errors(monkeypatch) -> None:
    """Provider exception messages must never echo credential strings."""
    monkeypatch.setattr(
        modal_account, "profile_credentials", lambda *_: ModalCredentials("profile", "ak-id", "as-secret")
    )
    monkeypatch.setattr(modal_account, "native_profiles", lambda: [])
    monkeypatch.setattr(modal_account, "verify_profile", AsyncMock(side_effect=RuntimeError("as-secret")))
    result = asyncio.run(ModalAccount().connect())
    assert result["status"] == "error"
    assert "as-secret" not in json.dumps(result)


def test_browser_login_uses_native_cli_and_only_exposes_modal_authorization_urls(monkeypatch) -> None:
    """Native login preserves the active account and rejects links on other hosts."""
    account = ModalAccount()
    observed = []
    options = {}

    class Process:
        def __init__(self):
            self.stdout = self.lines()

        def lines(self):
            for line in [
                "https://modal.com.evil.test/login\n",
                "https://modal.com/token-flow/test\n",
                "Enter this code: ABCD\n",
                "Token written to config in profile research.\n",
            ]:
                yield line
                observed.append(account.public())

        def wait(self):
            return 0

        def terminate(self):
            pass

    def spawn(command, **kwargs):
        options.update(command=command, **kwargs)
        return Process()

    monkeypatch.setattr(modal_account.subprocess, "Popen", spawn)
    monkeypatch.setattr(modal_account, "native_profiles", lambda: [])
    monkeypatch.setattr(
        modal_account, "profile_credentials", lambda *_: ModalCredentials("research", "ak-id", "as-secret")
    )
    monkeypatch.setattr(modal_account, "verify_profile", AsyncMock(return_value="Team"))
    account._status = "signing_in"
    account._login()
    assert options["command"][-3:] == ["token", "new", "--no-activate"]
    assert "MODAL_TOKEN_SECRET" not in options["env"]
    assert observed[0]["login_url"] == ""
    assert observed[1]["login_url"] == "https://modal.com/token-flow/test"
    assert observed[2]["code"] == "ABCD"
    assert account.public()["status"] == "connected"
    assert account.public()["login_url"] == ""


def test_profile_deployment_resolves_credentials_on_the_server(tmp_path: Path, monkeypatch) -> None:
    """A browser sends a profile name rather than Modal API credentials."""
    app = create_app(tmp_path)
    requested = []
    monkeypatch.setattr(app_runtime, "profile_credentials", lambda name: ModalCredentials(name, "ak-id", "as-secret"))
    monkeypatch.setattr(
        app.state.modal_deployments, "start", lambda *args, **kwargs: requested.append(args) or {"id": "deploy"}
    )
    response = TestClient(app).post("/api/modal/deploy", json={"profile": "research", "gpu": "L4"})
    assert response.status_code == 202
    assert requested == [("ak-id", "as-secret", "L4")]
    assert "as-secret" not in response.text


def test_run_deadline_cancels_and_releases_its_timer(tmp_path: Path, monkeypatch) -> None:
    """Deadline cancellation uses the same remote cancellation path as Stop."""
    manager = runtime.JobManager(tmp_path, restore_history=False)
    monkeypatch.setattr(manager, "_run_remote", lambda *_: None)
    job = manager.start(
        {
            "loader": "olsson",
            "name": "bounded",
            "execution_target": "remote",
            "remote_endpoint": "https://workspace.test",
            "api_key": "key",
            "max_run_minutes": 2,
        }
    )
    manager._expire_run(job["id"], 2)
    assert manager.get(job["id"]).status == "cancelled"
    assert "2-minute" in manager.get(job["id"]).error
    assert job["id"] not in manager._run_timers
    assert manager.get(job["id"]).remote_cancel_requested


def test_remote_upload_cache_skips_transfer_only_for_matching_contents(tmp_path: Path, monkeypatch) -> None:
    """An existing remote folder is reused before constructing an upload archive."""
    from visualization.datasets import dataset_fingerprint

    source = tmp_path / "input"
    source.mkdir()
    (source / "image.jpg").write_bytes(b"photo")
    cached = {"path": "/managed/input", "fingerprint": dataset_fingerprint(source)}
    monkeypatch.setattr(runtime.JobManager, "_remote_json", lambda *_: {"items": [cached]})
    assert (
        runtime.JobManager._remote_upload_directory("https://workspace", "key", source, tmp_path / "transfers")
        == cached
    )
    assert not (tmp_path / "transfers").exists()


def test_metric_summaries_keep_reprojection_quality_and_config_nonfinite_values(tmp_path: Path) -> None:
    """Nested metric summaries stay comparable and valid YAML infinity remains exportable."""
    root = tmp_path / "run"
    root.mkdir()
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "bundle_adjustment": {
                    "reprojection_error": {"summary": {"mean": 0.42, "median": 0.3}, "full_data": [0.1, 0.2]}
                }
            }
        )
    )
    (root / "config.yaml").write_text("threshold: .inf\n")
    details = experiments.experiment_details(_job(root), [])
    assert details["metrics"]["bundle_adjustment"] == {
        "reprojection_error/summary/mean": 0.42,
        "reprojection_error/summary/median": 0.3,
    }
    assert details["configuration"]["threshold"] == "inf"
    json.dumps(details, allow_nan=False)


def test_portable_import_rejects_non_scalar_pipeline_values() -> None:
    """Malformed input must fail clearly before reaching React or the runner."""
    for settings in [{"name": []}, {"hydra_overrides": {"model.foo": []}}, {"loader_options": []}]:
        with pytest.raises(ValueError):
            experiments.validate_import({"schema": experiments.SCHEMA, "settings": settings})


def test_source_preview_accepts_only_managed_uploads(tmp_path: Path) -> None:
    """Preview responses identify real managed images and reject traversal/symlinks."""
    from io import BytesIO

    from PIL import Image

    image = BytesIO()
    Image.new("RGB", (4, 4), "gold").save(image, "JPEG")
    client = TestClient(create_app(tmp_path))
    upload = client.post(
        "/api/uploads",
        data={"manifest": '["capture/one.jpg"]'},
        files={"files": ("one.jpg", image.getvalue(), "image/jpeg")},
    ).json()
    response = client.get(f'/api/uploads/{upload["id"]}/preview')
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert client.get(f'/api/uploads/{upload["id"]}/preview?index=3').status_code == 404
    assert client.get("/api/uploads/not-managed/preview").status_code == 404
    selected = Path(upload["path"]) / "one.jpg"
    selected.unlink()
    selected.symlink_to(tmp_path / "outside.jpg")
    assert client.get(f'/api/uploads/{upload["id"]}/preview').status_code == 404


def test_deploy_reuses_matching_source_and_protects_active_experiments(tmp_path: Path, monkeypatch) -> None:
    """A reusable workspace avoids redeployment; different code/GPU cannot disrupt active work."""
    from visualization import modal_deployment

    manager = modal_deployment.ModalDeploymentManager(lambda *_: {})
    monkeypatch.setattr(modal_deployment, "find_source_root", lambda: tmp_path)
    monkeypatch.setattr(modal_deployment, "source_identity", lambda *_: {"fingerprint": "same-code"})
    deployment = modal_deployment.ModalDeployment(
        id="known",
        gpu="L4",
        cpu=4,
        memory_mb=32768,
        status="completed",
        token_id="ak-test",
        token_secret="as-test",
        source={"fingerprint": "same-code"},
        api_key=modal_deployment.modal_workspace_api_key("ak-test", "as-test"),
    )
    deployment.token_id = deployment.token_secret = ""
    manager._deployments[deployment.id] = deployment
    assert manager.start("ak-test", "as-test", "L4", cpu=4, memory_mb=32768, allow_update=False)["id"] == "known"
    with pytest.raises(ValueError, match="Modal experiment is active"):
        manager.start("ak-test", "as-test", "L40S", allow_update=False)
    with pytest.raises(ValueError, match="Modal experiment is active"):
        manager.start("ak-test", "as-test", "L4", cpu=4, memory_mb=32768, allow_update=False, force=True)


def test_deployment_rejects_source_changes_during_image_preparation(tmp_path: Path, monkeypatch) -> None:
    """Source identity cannot silently describe files edited while a deployment was building."""
    from visualization import modal_deployment

    discovered = []
    manager = modal_deployment.ModalDeploymentManager(lambda *_: discovered.append(True) or {})
    deployment = modal_deployment.ModalDeployment(
        id="changing", gpu="L4", source={"commit": "test", "dirty": False, "fingerprint": "before"}
    )
    monkeypatch.setattr(modal_deployment, "source_identity", lambda *_: {"fingerprint": "after"})
    process = SimpleNamespace(stdout=iter([]), wait=lambda: 0)
    monkeypatch.setattr(modal_deployment.subprocess, "Popen", lambda *_args, **_kwargs: process)
    manager._run(deployment, tmp_path)
    assert deployment.status == "failed"
    assert "Source changed during setup" in deployment.error
    assert not discovered
