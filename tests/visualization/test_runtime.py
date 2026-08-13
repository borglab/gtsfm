"""Tests for the installable browser workspace runtime."""

from io import BytesIO
import json
from pathlib import Path
import tarfile
import threading

import numpy as np
import pytest
import spz
from fastapi.testclient import TestClient
from PIL import Image

from visualization import app as app_runtime
from visualization import modal_deployment
from visualization import runtime
from visualization.app import create_app
from visualization import samples as sample_runtime


def _cuda_hardware() -> dict:
    return {
        "devices": [
            {
                "id": "cpu",
                "kind": "cpu",
                "label": "CPU",
                "status": "available",
                "supports_gaussian_splatting": False,
            },
            {
                "id": "cuda:0",
                "kind": "cuda",
                "label": "Test GPU",
                "status": "available",
                "supports_gaussian_splatting": True,
            },
        ]
    }


def _write_test_splat(path: Path) -> None:
    """Write one valid Gaussian using the same official SPZ library as exports."""

    cloud = spz.GaussianCloud()
    cloud.positions = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cloud.scales = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cloud.rotations = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    cloud.alphas = np.array([0.0], dtype=np.float32)
    cloud.colors = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cloud.sh_degree = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    assert spz.save_splat_to_ply(cloud, spz.PackOptions(), str(path))


def test_configuration_schema_defaults_to_vggt() -> None:
    schema = runtime.configuration_schema()

    assert schema["defaults"]["config_name"] == "vggt"
    assert {item["id"] for item in schema["splat_implementations"]} == {"none", "gsplat", "anysplat"}
    assert "base_gs" in schema["gaussian_splatting_models"]
    models = {item["id"]: item for item in schema["models"]}
    assert models["vggt"]["capabilities"]["iterative_splat"] is True
    argoverse_options = {item["name"]: item for item in schema["loader_options"]["argoverse"]}
    assert argoverse_options["log_id"]["required"] is True


def test_detect_hardware_always_reports_cpu() -> None:
    hardware = runtime.detect_hardware()

    assert hardware["devices"]
    assert hardware["devices"][0]["id"] == "cpu"


def test_build_runner_args_for_vggt_gsplat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setattr(runtime, "detect_hardware", _cuda_hardware)

    args, env = runtime.build_runner_args(
        {
            "config_name": "vggt",
            "loader": "olsson",
            "dataset_dir": str(dataset),
            "hardware": "cuda:0",
            "splat_implementation": "gsplat",
            "gaussian_splatting_config_name": "base_gs",
            "gs_max_steps": 123,
        },
        tmp_path / "output",
    )

    assert args[:2] == ["--config_name", "vggt"]
    assert "--run_gs" in args
    assert args[args.index("--gs_max_steps") + 1] == "123"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_anysplat_selection_uses_anysplat_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setattr(runtime, "detect_hardware", _cuda_hardware)

    args, _ = runtime.build_runner_args(
        {
            "config_name": "vggt",
            "loader": "olsson",
            "dataset_dir": str(dataset),
            "hardware": "cuda:0",
            "splat_implementation": "anysplat",
        },
        tmp_path / "output",
    )

    assert args[:2] == ["--config_name", "anysplat"]
    assert "--run_gs" not in args


def test_loader_specific_options_are_validated_and_forwarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setattr(runtime, "detect_hardware", _cuda_hardware)
    base_spec = {
        "config_name": "vggt",
        "loader": "argoverse",
        "dataset_dir": str(dataset),
        "hardware": "cpu",
        "splat_implementation": "none",
    }

    with pytest.raises(ValueError, match="Log Id is required"):
        runtime.build_runner_args(base_spec, tmp_path / "missing-option")

    args, _ = runtime.build_runner_args(
        {**base_spec, "loader_options": {"log_id": "log 123", "stride": 2}},
        tmp_path / "output",
    )

    assert 'loader.log_id="log 123"' in args
    assert "loader.stride=2" in args


def test_workspace_api_and_scene_discovery(tmp_path: Path) -> None:
    scene = tmp_path / "example" / "ba_output"
    scene.mkdir(parents=True)
    (scene / "points3D.txt").write_text("# points\n", encoding="utf-8")
    (scene / "images.txt").write_text("# images\n", encoding="utf-8")
    client = TestClient(create_app(tmp_path))

    index_response = client.get("/")
    assert index_response.status_code == 200
    assert b"studio.js" in index_response.content
    assert b"bee-favicon.png" in index_response.content
    assert b"console.js" not in index_response.content
    assert client.get("/static/studio.js").status_code == 200
    assert client.get("/static/studio.css").status_code == 200
    assert client.get("/static/vendor/babylon.js").status_code == 200
    assert client.get("/static/brand/sfm-logo.png").status_code == 200
    assert client.get("/static/brand/bee-favicon.png").status_code == 200
    schema_response = client.get("/api/configuration")
    assert schema_response.status_code == 200
    assert schema_response.json()["defaults"]["config_name"] == "vggt"
    setup_response = client.get("/api/setup")
    assert setup_response.status_code == 200
    setup = setup_response.json()
    assert setup["status"] in {"ready", "warning", "error"}
    assert setup["summary"]
    assert setup["checked_at"]
    checks = {item["id"]: item for item in setup["items"]}
    assert checks["workspace"]["state"] == "ready"
    assert checks["submodule-vggt"]["required"] is True
    assert checks["submodule-fastvggt"]["required"] is False
    if checks["submodule-fastvggt"]["state"] == "optional":
        assert checks["submodule-fastvggt"]["action"]["label"] == "Download"
        assert checks["submodule-fastvggt"]["action"]["enabled"] is True
    if checks["gaussian-splatting"]["state"] == "optional":
        assert checks["gaussian-splatting"]["action"]["enabled"] is False
    scenes_response = client.get("/api/scenes")
    assert scenes_response.status_code == 200
    assert scenes_response.json()["count"] == 1
    assert scenes_response.json()["items"][0]["kind"] == "scene"

    samples_response = client.get("/api/samples")
    assert samples_response.status_code == 200
    samples = {item["id"]: item for item in samples_response.json()["items"]}
    assert set(samples) == {"lund-door", "crane-mast", "mobilebrick"}
    assert samples["crane-mast"]["recommendations"]["loader"] == "colmap"
    assert samples["lund-door"]["source_url"].startswith("https://github.com/borglab/gtsfm/")


def test_workspace_exports_saved_splat_as_ply_and_spz(tmp_path: Path) -> None:
    source = tmp_path / "example" / "gaussian_splats.ply"
    _write_test_splat(source)
    client = TestClient(create_app(tmp_path))

    ply_response = client.get("/api/splats/export", params={"path": "example/gaussian_splats.ply", "format": "ply"})
    assert ply_response.status_code == 200
    assert '.ply"' in ply_response.headers["content-disposition"]
    assert ply_response.content.startswith(b"ply\n")

    spz_response = client.get("/api/splats/export", params={"path": "example/gaussian_splats.ply", "format": "spz"})
    assert spz_response.status_code == 200
    assert '.spz"' in spz_response.headers["content-disposition"]
    assert spz_response.content.startswith(b"NGSP")
    exported = tmp_path / "exported.spz"
    exported.write_bytes(spz_response.content)
    assert spz.load_spz(str(exported), spz.UnpackOptions()).num_points == 1

    cached_response = client.get("/api/splats/export", params={"path": "example/gaussian_splats.ply", "format": "spz"})
    assert cached_response.content == spz_response.content


def test_completed_job_exposes_splat_download(tmp_path: Path) -> None:
    output_root = tmp_path / "runs" / "finished"
    _write_test_splat(output_root / "gaussian_splats.ply")
    app = create_app(tmp_path)
    manager = app.state.gtsfm_jobs
    manager._jobs["finished-job"] = runtime.ManagedJob(
        id="finished-job",
        name="finished-scene",
        status="completed",
        created_at="2026-08-13T00:00:00+00:00",
        updated_at="2026-08-13T00:00:00+00:00",
        output_root=str(output_root),
        live_root=str(manager.live_root("finished-job")),
        spec={},
        command=[],
    )
    client = TestClient(app)

    jobs = client.get("/api/jobs").json()["items"]
    assert jobs[0]["has_final_splat"] is True
    response = client.get("/api/jobs/finished-job/splat", params={"format": "spz"})
    assert response.status_code == 200
    assert response.content.startswith(b"NGSP")


def test_live_job_exposes_dask_worker_status(tmp_path: Path) -> None:
    app = create_app(tmp_path)
    manager = app.state.gtsfm_jobs
    live_root = manager.live_root("running-job")
    live_root.mkdir(parents=True)
    (live_root / "dask.json").write_text(
        json.dumps(
            {
                "workers": 2,
                "threads": 8,
                "running_tasks": 3,
                "completed_tasks": 14,
                "pending_tasks": 5,
                "failed_tasks": 0,
                "memory_bytes": 1024,
                "memory_limit_bytes": 4096,
                "cpu_percent": 71.5,
            }
        ),
        encoding="utf-8",
    )
    manager._jobs["running-job"] = runtime.ManagedJob(
        id="running-job",
        name="live-scene",
        status="running",
        created_at="2026-08-13T00:00:00+00:00",
        updated_at="2026-08-13T00:00:00+00:00",
        output_root=str(tmp_path / "runs" / "live-scene"),
        live_root=str(live_root),
        spec={},
        command=[],
    )

    response = TestClient(app).get("/api/jobs/running-job/live")

    assert response.status_code == 200
    assert response.json()["dask"]["workers"] == 2
    assert response.json()["dask"]["running_tasks"] == 3


def test_workspace_rejects_invalid_splat_exports(tmp_path: Path) -> None:
    source = tmp_path / "gaussian_splats.ply"
    _write_test_splat(source)
    client = TestClient(create_app(tmp_path))

    unsupported = client.get("/api/splats/export", params={"path": source.name, "format": "obj"})
    assert unsupported.status_code == 400
    traversal = client.get("/api/splats/export", params={"path": "../gaussian_splats.ply", "format": "ply"})
    assert traversal.status_code == 403


def test_workspace_prepares_local_github_sample(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).post("/api/samples/lund-door/prepare")

    assert response.status_code == 200
    assert Path(response.json()["path"]).name == "set1_lund_door"
    assert Path(response.json()["path"]).is_dir()
    assert response.json()["sample"]["recommendations"]["config_name"] == "vggt"


def test_sample_downloader_uses_certifi_ca_bundle() -> None:
    assert sample_runtime._SSL_CONTEXT.get_ca_certs()


def test_workspace_rejects_unknown_github_sample(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).post("/api/samples/not-a-sample/prepare")

    assert response.status_code == 404


def test_workspace_rejects_invalid_run(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).post(
        "/api/jobs",
        json={
            "config_name": "vggt",
            "loader": "olsson",
            "dataset_dir": str(tmp_path / "missing"),
            "hardware": "cpu",
            "splat_implementation": "none",
        },
    )

    assert response.status_code == 400
    assert "does not exist" in response.json()["error"]


def test_workspace_discovers_modal_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_discovery(token_id: str, token_secret: str) -> dict[str, str]:
        assert token_id == "ak-test"
        assert token_secret == "as-test"
        return {
            "endpoint": "https://example--gtsfm-workspace.modal.run",
            "app_name": "gtsfm",
            "function_name": "workspace",
        }

    monkeypatch.setattr(app_runtime, "_discover_modal_endpoint", fake_discovery)

    response = TestClient(create_app(tmp_path)).post(
        "/api/modal/discover",
        json={"token_id": "ak-test", "token_secret": "as-test"},
    )

    assert response.status_code == 200
    assert response.json()["endpoint"] == "https://example--gtsfm-workspace.modal.run"
    assert response.json()["api_key"].startswith("gtsfm_")


def test_workspace_starts_and_reports_modal_deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(tmp_path)
    manager = app.state.modal_deployments
    requested: dict[str, object] = {}
    started = {
        "id": "deploy-1",
        "gpu": "L40S",
        "cpu": 12,
        "memory_mb": 98304,
        "status": "queued",
        "stage": "Waiting to deploy",
        "log_tail": [],
        "endpoint": "",
        "api_key": "",
        "error": "",
    }

    def fake_start(token_id: str, token_secret: str, gpu: str, **resources: object) -> dict[str, object]:
        requested.update(resources)
        return started

    monkeypatch.setattr(manager, "start", fake_start)
    monkeypatch.setattr(manager, "get", lambda deployment_id: {**started, "status": "running", "stage": "Building"})
    monkeypatch.setattr(manager, "cancel", lambda deployment_id: {**started, "status": "cancelling"})
    client = TestClient(app)

    response = client.post(
        "/api/modal/deploy",
        json={
            "token_id": "ak-test",
            "token_secret": "as-test",
            "gpu": "L40S",
            "cpu": 12,
            "memory_mb": 98304,
        },
    )
    status = client.get("/api/modal/deploy/deploy-1")
    cancelled = client.post("/api/modal/deploy/deploy-1/cancel")

    assert response.status_code == 202
    assert response.json()["id"] == "deploy-1"
    assert requested == {"cpu": 12, "memory_mb": 98304}
    assert status.status_code == 200
    assert status.json()["stage"] == "Building"
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelling"


def test_workspace_rejects_unknown_setup_action(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).post("/api/setup/not-installable/install")

    assert response.status_code == 400
    assert "does not have an automatic action" in response.json()["error"]


def test_optional_submodule_setup_downloads_and_rechecks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_root = tmp_path / "site-packages" / "gtsfm"
    package_root.mkdir(parents=True)
    results_root = tmp_path / "results"
    results_root.mkdir()
    monkeypatch.setattr(runtime, "PACKAGE_ROOT", package_root)
    original_which = runtime.shutil.which
    monkeypatch.setattr(
        runtime.shutil,
        "which",
        lambda command: "/usr/bin/git" if command == "git" else original_which(command),
    )

    def fake_clone(command: list[str], **_kwargs: object) -> None:
        destination = Path(command[-1])
        destination.mkdir(parents=True)
        (destination / "README.md").write_text("FastVGGT", encoding="utf-8")

    monkeypatch.setattr(runtime, "_run_setup_command", fake_clone)

    payload = runtime.install_optional_setup("submodule-fastvggt", results_root)

    assert payload["item"]["state"] == "ready"
    assert (package_root.parent / "thirdparty" / "FastVGGT" / "README.md").is_file()


def test_workspace_imports_dropped_folder(tmp_path: Path) -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (1200, 800), color=(25, 50, 75)).save(image_buffer, format="JPEG")
    response = TestClient(create_app(tmp_path)).post(
        "/api/uploads",
        data={"manifest": json.dumps(["example/images/one.jpg", "example/cameras.txt"])},
        files=[
            ("files", ("one.jpg", BytesIO(image_buffer.getvalue()), "image/jpeg")),
            ("files", ("cameras.txt", BytesIO(b"camera"), "text/plain")),
        ],
    )

    assert response.status_code == 200
    imported = Path(response.json()["path"])
    assert response.json()["name"] == "example"
    assert response.json()["file_count"] == 2
    assert response.json()["analysis"]["image_count"] == 1
    assert response.json()["analysis"]["total_megapixels"] == 0.96
    assert response.json()["analysis"]["max_width"] == 1200
    assert imported.is_dir()
    assert (imported / "images" / "one.jpg").read_bytes() == image_buffer.getvalue()
    assert (imported / "cameras.txt").read_bytes() == b"camera"


def test_workspace_rejects_unsafe_upload_path(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).post(
        "/api/uploads",
        data={"manifest": json.dumps(["../outside.txt"])},
        files=[("files", ("outside.txt", BytesIO(b"nope"), "text/plain"))],
    )

    assert response.status_code == 400
    assert "Unsafe path" in response.json()["error"]
    assert not (tmp_path.parent / "outside.txt").exists()


def _tar_upload(entries: dict[str, bytes]) -> BytesIO:
    payload = BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, BytesIO(content))
    payload.seek(0)
    return payload


def test_workspace_imports_remote_folder_archive(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).post(
        "/api/uploads/archive",
        files={
            "archive": (
                "dataset.tar",
                _tar_upload({"images/one.jpg": b"image", "cameras.txt": b"camera"}),
                "application/x-tar",
            )
        },
    )

    assert response.status_code == 200
    imported = Path(response.json()["path"])
    assert response.json()["file_count"] == 2
    assert (imported / "images" / "one.jpg").read_bytes() == b"image"
    assert (imported / "cameras.txt").read_bytes() == b"camera"


def test_workspace_rejects_unsafe_remote_archive(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).post(
        "/api/uploads/archive",
        files={"archive": ("dataset.tar", _tar_upload({"../outside.txt": b"nope"}), "application/x-tar")},
    )

    assert response.status_code == 400
    assert "Unsafe path" in response.json()["error"]
    assert not (tmp_path.parent / "outside.txt").exists()


def test_remote_job_keeps_api_key_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(threading.Thread, "start", lambda _thread: None)
    manager = runtime.JobManager(tmp_path)

    job = manager.start(
        {
            "name": "remote-scene",
            "execution_target": "remote",
            "remote_endpoint": "https://gpu.example.test",
            "api_key": "super-secret",
            "config_name": "vggt",
        }
    )

    assert "api_key" not in job["spec"]
    assert "super-secret" not in str(job)
    assert job["remote"]["endpoint"] == "https://gpu.example.test"


def test_job_cancel_terminates_local_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        def poll(self) -> None:
            return None

    manager = runtime.JobManager(tmp_path)
    job = runtime.ManagedJob(
        id="local-job",
        name="local",
        status="running",
        created_at="now",
        updated_at="now",
        output_root=str(tmp_path / "output"),
        live_root=str(tmp_path / "live"),
        spec={},
        command=[],
    )
    process = FakeProcess()
    job.process = process  # type: ignore[assignment]
    manager._jobs[job.id] = job
    terminated: list[object] = []
    monkeypatch.setattr(manager, "_terminate_process", terminated.append)

    cancelled = manager.cancel(job.id)

    assert cancelled["status"] == "cancelled"
    assert terminated == [process]


def test_job_cancel_forwards_to_remote_vm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = runtime.JobManager(tmp_path)
    job = runtime.ManagedJob(
        id="remote-job",
        name="remote",
        status="running",
        created_at="now",
        updated_at="now",
        output_root=str(tmp_path / "output"),
        live_root=str(tmp_path / "live"),
        spec={},
        command=[],
        remote={"endpoint": "https://gpu.example.test", "job_id": "upstream-job"},
        remote_api_key="secret",
    )
    manager._jobs[job.id] = job
    calls: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        manager,
        "_remote_json",
        lambda endpoint, api_key, payload=None: calls.append((endpoint, api_key, payload)) or {},
    )

    cancelled = manager.cancel(job.id)

    assert cancelled["status"] == "cancelled"
    assert calls == [("https://gpu.example.test/api/jobs/upstream-job/cancel", "secret", {})]


def test_remote_job_keeps_modal_and_ssh_credentials_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(threading.Thread, "start", lambda _thread: None)
    manager = runtime.JobManager(tmp_path)

    job = manager.start(
        {
            "name": "modal-scene",
            "execution_target": "remote",
            "remote_endpoint": "https://example--gtsfm.modal.run",
            "api_key": "ak-visible-only-in-memory.as-secret",
            "modal_api_key": "gtsfm-derived-secret",
            "modal_token_id": "ak-visible-only-in-memory",
            "modal_token_secret": "as-secret",
            "modal_token_command": "modal token set --token-id ak-visible-only-in-memory --token-secret as-secret",
            "ssh_private_key": "/private/key/path",
            "config_name": "vggt",
        }
    )

    for field in (
        "api_key",
        "modal_api_key",
        "modal_token_id",
        "modal_token_secret",
        "modal_token_command",
        "ssh_private_key",
    ):
        assert field not in job["spec"]
    assert "as-secret" not in str(job)
    assert "/private/key/path" not in str(job)


def test_remote_job_transfers_local_dataset_before_submit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "image.jpg").write_bytes(b"image")
    monkeypatch.setattr(threading.Thread, "start", lambda _thread: None)
    monkeypatch.setattr(runtime.time, "sleep", lambda _seconds: None)
    posted: dict[str, object] = {}

    def fake_upload(_endpoint: str, _api_key: str, directory: Path, _runtime_root: Path) -> dict[str, object]:
        assert directory == dataset
        return {"path": "/workspace/results/.gtsfm/uploads/remote-dataset"}

    def fake_remote_json(url: str, _api_key: str, *, payload: dict[str, object] | None = None) -> dict[str, object]:
        if url.endswith("/api/jobs"):
            assert payload is not None
            posted.update(payload)
            return {"id": "remote-job", "status": "queued"}
        if url.endswith("/live"):
            return {}
        return {"id": "remote-job", "status": "completed", "return_code": 0, "log_tail": []}

    monkeypatch.setattr(runtime.JobManager, "_remote_upload_directory", staticmethod(fake_upload))
    monkeypatch.setattr(runtime.JobManager, "_remote_json", staticmethod(fake_remote_json))
    manager = runtime.JobManager(tmp_path / "results")
    public = manager.start(
        {
            "name": "remote-transfer",
            "execution_target": "remote",
            "remote_endpoint": "https://gpu.example.test",
            "api_key": "secret",
            "dataset_dir": str(dataset),
            "config_name": "vggt",
            "splat_implementation": "none",
        }
    )
    managed = manager.get(public["id"])
    assert managed is not None

    manager._run_remote(managed)

    assert posted["dataset_dir"] == "/workspace/results/.gtsfm/uploads/remote-dataset"
    assert posted["execution_target"] == "local"
    assert managed.status == "completed"


def test_remote_sample_is_downloaded_by_remote_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_runtime,
        "prepare_sample",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sample should not download locally")),
    )
    app = create_app(tmp_path)
    manager = app.state.gtsfm_jobs
    monkeypatch.setattr(
        manager,
        "start",
        lambda spec: {"id": "remote-sample", "status": "queued", "spec": dict(spec), "remote": {}},
    )

    response = TestClient(app).post(
        "/api/jobs",
        json={
            "name": "remote-sample",
            "sample_id": "lund-door",
            "execution_target": "remote",
            "remote_endpoint": "https://example--gtsfm.modal.run",
            "api_key": "secret",
            "loader": "olsson",
            "config_name": "vggt",
            "splat_implementation": "none",
        },
    )

    assert response.status_code == 202
    assert response.json()["spec"]["sample_id"] == "lund-door"


def test_modal_remote_api_key_is_stable_and_scoped() -> None:
    first = modal_deployment.modal_remote_api_key("ak-one", "as-secret")

    assert first == modal_deployment.modal_remote_api_key("ak-one", "as-secret")
    assert first != modal_deployment.modal_remote_api_key("ak-two", "as-secret")
    assert "as-secret" not in first


def test_modal_deployment_rejects_invalid_machine_resources() -> None:
    manager = modal_deployment.ModalDeploymentManager(lambda *_args: {})

    with pytest.raises(ValueError, match="CPU allocation"):
        manager.start("ak-test", "as-test", "L40S", cpu=64, memory_mb=65536)
    with pytest.raises(ValueError, match="memory allocation"):
        manager.start("ak-test", "as-test", "L40S", cpu=8, memory_mb=1024)


def test_modal_deployment_prefers_prebuilt_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = modal_deployment.ModalDeploymentManager(
        lambda *_args: {},
        runtime_image="ghcr.io/borglab/gtsfm-modal-runtime:test",
    )
    monkeypatch.setattr(modal_deployment.threading.Thread, "start", lambda _thread: None)

    deployment = manager.start("ak-test", "as-test", "L4", cpu=4, memory_mb=32768)

    assert deployment["image_source"] == "prebuilt"
    assert deployment["runtime_image"].endswith(":test")


def test_modal_deployment_cancel_marks_active_setup_and_terminates_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = modal_deployment.ModalDeploymentManager(lambda *_args: {})
    deployment = modal_deployment.ModalDeployment(id="deploy-1", gpu="L4", status="running")
    process = object()
    deployment.process = process  # type: ignore[assignment]
    manager._deployments[deployment.id] = deployment
    terminated: list[object] = []
    monkeypatch.setattr(manager, "_terminate_process", terminated.append)

    cancelled = manager.cancel(deployment.id)

    assert cancelled["status"] == "cancelling"
    assert cancelled["stage"] == "Stopping Modal workspace setup"
    assert terminated == [process]


def test_remote_inspection_rejects_non_http_url(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).post(
        "/api/remote/inspect",
        json={"endpoint": "file:///etc", "api_key": "secret"},
    )

    assert response.status_code == 400


def test_fastapi_exposes_openapi_and_job_websocket(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))

    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert schema.json()["info"]["title"] == "GTSFM Studio API"

    with client.websocket_connect("/api/events/jobs") as websocket:
        assert websocket.receive_json() == {"items": []}
