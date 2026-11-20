"""Acceptance checks for saved settings, local Dask workers and remote result retrieval."""

import json
import tarfile
import threading
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from visualization.app import create_app
from visualization.execution import ExecutionConfiguration, ExecutionStore
from visualization.runtime import JobManager, ManagedJob


def configuration(name="Cyber", **settings):
    return ExecutionConfiguration(name=name, settings=settings)


def test_saved_settings_survive_restart_and_edits_are_explicit(tmp_path):
    path = tmp_path / "configurations.json"
    store = ExecutionStore(path)
    store.save(configuration(hardware="cuda:0", num_workers=3, threads_per_worker=2, worker_memory_limit="6GiB"))
    loaded = ExecutionStore(path).list()[0]
    assert loaded["settings"]["num_workers"] == 3
    assert loaded["settings"]["threads_per_worker"] == 2
    loaded["settings"]["num_workers"] = 7
    assert store.list()[0]["settings"]["num_workers"] == 3
    store.save(ExecutionConfiguration.model_validate(loaded))
    assert store.list()[0]["settings"]["num_workers"] == 7
    assert path.stat().st_mode & 0o777 == 0o600
    store.delete("Cyber")
    assert store.list() == []


@pytest.mark.parametrize(
    "settings",
    [
        {"api_key": "secret"},
        {"modal_token_secret": "secret"},
        {"num_workers": 0},
        {"cluster_config": "hidden.yaml"},
    ],
)
def test_store_rejects_credentials_invalid_resources_and_hidden_yaml(settings):
    with pytest.raises(ValidationError):
        configuration(**settings)


def test_corrupt_store_is_reported_without_overwrite(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("broken")
    with pytest.raises(ValueError):
        ExecutionStore(path).save(configuration())
    assert path.read_text() == "broken"


def test_legacy_ray_presets_do_not_hide_or_overwrite_supported_presets(tmp_path):
    path = tmp_path / "configurations.json"
    old_ray = {"name": "Remote workstation", "settings": {"remote_connection": "ssh", "ssh_host": "workstation"}}
    old_modal = {
        "name": "Cloud",
        "settings": {"execution_target": "remote", "remote_connection": "api", "ssh_port": 22, "modal_gpu": "L4"},
    }
    path.write_text(json.dumps([old_ray, old_modal]))
    store = ExecutionStore(path)

    rows = store.list()
    assert [row["name"] for row in rows] == ["Cloud"]
    assert rows[0]["settings"]["modal_gpu"] == "L4"
    store.save(configuration("CPU"))
    assert old_ray in json.loads(path.read_text())


def test_configuration_api_is_local_and_authenticated(tmp_path, monkeypatch):
    monkeypatch.setenv("GTSFM_API_KEY", "test-key")
    path = tmp_path / "on-mac" / "config.json"
    app = create_app(tmp_path / "results", execution_store=ExecutionStore(path))
    with TestClient(app) as client:
        assert client.get("/api/execution-configurations").status_code == 401
        client.headers["Authorization"] = "Bearer test-key"
        saved = client.put("/api/execution-configurations", json=configuration(hardware="cuda:0").model_dump())
        assert saved.status_code == 200
        assert client.get("/api/execution-configurations").json()["items"] == [saved.json()]
        assert json.loads(path.read_text())[0]["settings"]["hardware"] == "cuda:0"
        assert client.delete("/api/execution-configurations/Cyber").status_code == 200
        assert client.get("/api/execution-configurations").json()["items"] == []


def test_cluster_controls_generate_native_dask_constructor(tmp_path, monkeypatch):
    from visualization import runtime

    manager = JobManager(tmp_path)
    seen = []
    done = threading.Event()

    def build(spec, output):
        return [], {"CUDA_VISIBLE_DEVICES": "1", "GTSFM_SELECTED_DEVICE": "cuda:1"}

    def run(job, env):
        seen.append(yaml.safe_load(Path(job.command[job.command.index("--cluster_config") + 1]).read_text()))
        done.set()

    monkeypatch.setattr(runtime, "build_runner_args", build)
    monkeypatch.setattr(manager, "_run", run)
    manager.start(
        {
            "dask_worker_hosts": "node1 node2",
            "dask_scheduler_host": "head",
            "dask_worker_python": "/env/bin/python",
            "num_workers": 2,
            "threads_per_worker": 3,
            "worker_memory_limit": "7GiB",
            "dask_tmpdir": "/scratch",
        }
    )
    assert done.wait(2)
    assert seen == [
        {
            "_target_": "distributed.SSHCluster",
            "hosts": ["head", "node1", "node2"],
            "remote_python": "/env/bin/python",
            "scheduler_options": {"dashboard_address": "127.0.0.1:0", "port": 0},
            "worker_options": {
                "n_workers": 2,
                "nthreads": 3,
                "memory_limit": "7GiB",
                "local_directory": "/scratch",
                "env": {"CUDA_VISIBLE_DEVICES": "1", "GTSFM_SELECTED_DEVICE": "cuda:1"},
            },
        }
    ]


def test_complete_reconstruction_archive_available_without_splats(tmp_path):
    manager = JobManager(tmp_path)
    output = tmp_path / "runs" / "example"
    output.mkdir(parents=True)
    (output / "cameras.txt").write_text("complete camera data")
    (output / "points3D.txt").write_text("complete point data")
    outside = tmp_path / "private-key"
    outside.write_text("secret")
    (output / "bad-link").symlink_to(outside)
    manager._jobs["example"] = ManagedJob(
        id="example",
        name="example",
        status="completed",
        created_at="now",
        updated_at="now",
        output_root=str(output),
        live_root=str(tmp_path / "live"),
        spec={},
        command=[],
    )
    with TestClient(create_app(tmp_path, job_manager=manager)) as client:
        response = client.get("/api/jobs/example/results")
        assert response.status_code == 200
        import io

        with tarfile.open(fileobj=io.BytesIO(response.content)) as archive:
            assert set(archive.getnames()) == {"cameras.txt", "points3D.txt"}
            assert archive.extractfile("points3D.txt").read() == b"complete point data"
        manager.get("example").status = "running"
        assert client.get("/api/jobs/example/results").status_code == 409


def test_cancel_during_result_transfer_cannot_become_completed(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    job = ManagedJob(
        id="transfer",
        name="transfer",
        status="running",
        created_at="now",
        updated_at="now",
        output_root=str(tmp_path / "output"),
        live_root=str(tmp_path / "live"),
        spec={"splat_implementation": "gsplat"},
        command=[],
        remote={"endpoint": "https://example.test"},
        remote_api_key="secret",
    )
    manager._jobs[job.id] = job
    monkeypatch.setattr(manager, "_remote_json", lambda *a, **kw: {"id": "upstream", "status": "completed"})
    monkeypatch.setattr(manager, "_remote_download", lambda *a: manager.cancel(job.id))
    manager._run_remote(job)
    assert job.status == "cancelled"


def test_shutdown_does_not_cancel_unowned_restored_cloud_calls(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    manager._jobs["cloud"] = ManagedJob(
        id="cloud",
        name="cloud",
        status="running",
        created_at="now",
        updated_at="now",
        output_root=str(tmp_path),
        live_root=str(tmp_path),
        spec={},
        command=["modal"],
    )
    monkeypatch.setattr(manager, "cancel", lambda *a: pytest.fail("cancelled an unowned cloud call"))
    manager.shutdown()
    assert manager.get("cloud").status == "running"


def test_cpu_control_node_can_bind_a_gpu_on_ssh_workers(tmp_path, monkeypatch):
    from visualization import runtime

    monkeypatch.setattr(runtime, "detect_hardware", lambda: pytest.fail("queried the control node for worker hardware"))
    _, env = runtime.build_runner_args(
        {
            "dataset_dir": str(tmp_path),
            "loader": "olsson",
            "config_name": "vggt_sift_frontend_megaloc",
            "splat_implementation": "none",
            "dask_worker_hosts": "gpu-worker",
            "hardware": "cuda:1",
        },
        tmp_path / "out",
    )
    assert env == {"CUDA_VISIBLE_DEVICES": "1", "GTSFM_SELECTED_DEVICE": "cuda:1"}
