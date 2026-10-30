"""Acceptance checks for locally saved execution settings and SSH job routing."""

import json
import shutil
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from visualization.app import create_app
from visualization.execution import ExecutionConfiguration, ExecutionStore
from visualization.ray_backend import RayWorkspace
from visualization.runtime import JobManager, ManagedJob, configuration_schema


def configuration(name="Cyber", **settings):
    return ExecutionConfiguration(name=name, settings=settings)


def test_cluster_constructor_is_not_offered_as_reconstruction_model():
    assert "cluster_local" not in {item["id"] for item in configuration_schema()["models"]}


def test_saved_settings_survive_restart_and_edits_are_explicit(tmp_path):
    path = tmp_path / "configurations.json"
    store = ExecutionStore(path)
    store.save(configuration(ssh_host="cyber", num_workers=3, threads_per_worker=2, worker_memory_limit="6GiB"))
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
        {"ssh_port": 70000},
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


def test_concurrent_saves_keep_every_named_configuration(tmp_path):
    store = ExecutionStore(tmp_path / "config.json")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: store.save(configuration(name=f"Machine {n}")), range(12)))
    assert len(store.list()) == 12
    assert not list(tmp_path.glob("*.tmp"))


def test_configuration_api_is_local_and_authenticated(tmp_path, monkeypatch):
    monkeypatch.setenv("GTSFM_API_KEY", "test-key")
    path = tmp_path / "on-mac" / "config.json"
    app = create_app(tmp_path / "results", execution_store=ExecutionStore(path))
    with TestClient(app) as client:
        assert client.get("/api/execution-configurations").status_code == 401
        client.headers["Authorization"] = "Bearer test-key"
        saved = client.put("/api/execution-configurations", json=configuration(ssh_host="cyber").model_dump())
        assert saved.status_code == 200
        assert client.get("/api/execution-configurations").json()["items"] == [saved.json()]
        assert json.loads(path.read_text())[0]["settings"]["ssh_host"] == "cyber"
        assert client.delete("/api/execution-configurations/Cyber").status_code == 200
        assert client.get("/api/execution-configurations").json()["items"] == []


@pytest.mark.parametrize("host", ["-oProxyCommand=bad", "host;touch x", "host\ncommand", "$(bad)"])
def test_ssh_rejects_option_and_shell_injection(host):
    with pytest.raises(ValueError):
        RayWorkspace({"ssh_host": host, "ssh_username": "user"}, local_ray_binary="/tools/ray")


def test_ray_preserves_alias_and_uses_native_agent_configuration(tmp_path):
    workspace = RayWorkspace({"ssh_host": "cyber", "ssh_username": "user"}, tmp_path, local_ray_binary="/tools/ray")
    try:
        assert workspace.configuration["provider"]["head_ip"] == "cyber"
        assert workspace.configuration["auth"] == {"ssh_user": "user"}
    finally:
        workspace.close()


def test_ssh_job_uses_connector_without_local_runner(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    seen = []
    event = threading.Event()

    def run_ssh(job):
        seen.append(job.ssh_spec)
        event.set()

    monkeypatch.setattr(manager, "_run_ssh", run_ssh)
    spec = {
        "execution_target": "remote",
        "remote_connection": "ssh",
        "ssh_host": "cyber",
        "ssh_username": "user",
        "ssh_workspace": "/opt/gtsfm",
        "ssh_private_key": "/secret/key",
        "loader": "olsson",
    }
    result = manager.start(spec)
    assert event.wait(2)
    assert seen == [spec]
    assert result["command"] == ["ray", "job", "submit", "cyber"]
    assert "ssh_private_key" not in result["spec"]
    assert manager.get(result["id"]).process is None


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


def test_shutdown_closes_ray_workspace_during_startup_and_joins_thread(tmp_path, monkeypatch):
    from visualization import runtime

    started, closed = threading.Event(), threading.Event()

    class Workspace:
        stderr_tail = ""

        def __init__(self, spec, runtime_root):
            self.entered = False

        def __enter__(self):
            self.entered = True
            started.set()
            assert closed.wait(5)
            raise RuntimeError("native launcher closed during authentication")

        def __exit__(self, *args):
            self.close()

        def close(self):
            if self.entered:
                closed.set()

    monkeypatch.setattr(runtime, "RayWorkspace", Workspace)
    manager = JobManager(tmp_path)
    created = manager.start(
        {"execution_target": "remote", "remote_connection": "ssh", "ssh_host": "cyber", "ssh_workspace": "/opt/gtsfm"}
    )
    assert started.wait(2)
    manager.shutdown()
    assert closed.is_set()
    assert not any(thread.is_alive() for thread in manager._threads)
    assert manager.get(created["id"]).status == "cancelled"
    assert not manager._ssh_workspaces


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


@pytest.fixture
def ray_execution(tmp_path, monkeypatch):
    """Exercise real job orchestration and local files with a fake native Ray adapter."""
    from visualization import runtime

    source = tmp_path / "photos on Mac"
    (source / "images").mkdir(parents=True)
    Image.new("RGB", (32, 24), "red").save(source / "images" / "camera.jpg")
    (source / "metadata.json").write_text('{"source":"mac"}')
    remote = tmp_path / "remote"
    uploads, submissions, stops, sessions = [], [], [], []
    options = {"uncertain_submit": False, "cancel_download": False}
    manager = JobManager(tmp_path / "results")

    class Workspace:
        stderr_tail = ""

        def __init__(self, spec, runtime_root):
            self.owned = []
            self.closed = False

        def __enter__(self):
            sessions.append(self)
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            if not self.closed:
                self.closed = True
                stops.extend(self.owned)

        def upload_directory(self, local, destination):
            assert destination.startswith("/opt/gtsfm/")
            target = remote / destination.removeprefix("/opt/gtsfm/")
            shutil.copytree(local, target, dirs_exist_ok=True)
            uploads.append((local, destination, target))

        def submit(self, command, **kwargs):
            submissions.append((command, kwargs))
            self.owned.append(kwargs["submission_id"])
            if options["uncertain_submit"]:
                raise ConnectionError("submission response lost after acceptance")
            output = remote / command[command.index("--output_root") + 1]
            scene = output / "results" / "merged_retriangulated"
            scene.mkdir(parents=True)
            (scene / "images.txt").write_text("0 1 0 0 0 0 0 0 0 camera.jpg\n\n")
            (scene / "cameras.txt").write_text("0 PINHOLE 32 24 20 20 16 12\n")
            (scene / "points3D.txt").write_text("# no points in this transport fixture\n")
            (output / "config.yaml").write_text(
                yaml.safe_dump({"loader": {"dataset_dir": command[command.index("--dataset_dir") + 1]}})
            )
            return kwargs["submission_id"]

        def get(self, job_id):
            return SimpleNamespace(status="SUCCEEDED", message="")

        def logs(self, job_id):
            return "Ray-supervised pipeline completed\n"

        async def tail_logs(self, job_id):
            yield self.logs(job_id)

        def download_directory(self, source, destination):
            shutil.copytree(remote / source.removeprefix("/opt/gtsfm/"), destination, dirs_exist_ok=True)
            if options["cancel_download"]:
                manager.cancel("ray-flow")

    monkeypatch.setattr(runtime, "RayWorkspace", Workspace)
    catalog = {
        "cpu_count": 8,
        "hardware": {
            "devices": [
                {"id": "cpu", "status": "available", "supports_gaussian_splatting": False},
                {"id": "cuda:0", "status": "available", "supports_gaussian_splatting": True},
            ]
        },
    }
    monkeypatch.setattr(manager, "_inspect_ray", lambda *args: catalog)
    spec = {
        "execution_target": "remote",
        "remote_connection": "ssh",
        "ssh_host": "gpu-linux",
        "ssh_username": "researcher",
        "ssh_workspace": "/opt/gtsfm",
        "ssh_python": "/envs/gtsfm/bin/python",
        "dataset_dir": str(source),
        "config_name": "vggt_sift_frontend_megaloc",
        "loader": "olsson",
        "remote_hardware": "cuda:0",
        "num_workers": 3,
        "threads_per_worker": 2,
        "worker_memory_limit": "7GiB",
        "correspondence_generator_config_name": "sift_brute_force",
        "live_preview_interval": 150,
    }
    yield SimpleNamespace(
        manager=manager,
        spec=spec,
        source=source,
        uploads=uploads,
        submissions=submissions,
        stops=stops,
        sessions=sessions,
        options=options,
    )
    manager.shutdown()


def _finish_ray_flow(case):
    """Start and join the real orchestration thread without remote computation."""
    created = case.manager.start(case.spec, job_id="ray-flow")
    for thread in case.manager._threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    return case.manager.get(created["id"])


def test_ray_stages_real_inputs_submits_saved_settings_and_serves_downloaded_photos(ray_execution):
    """Verify the full application transfer path through results, photos, and archive APIs."""
    case = ray_execution
    job = _finish_ray_flow(case)
    assert job.status == "completed", job.error
    local, remote, staged = case.uploads[0]
    assert local == case.source
    assert remote == "/opt/gtsfm/.gtsfm/staged/ray-flow/dataset_dir"
    assert (staged / "metadata.json").read_bytes() == (case.source / "metadata.json").read_bytes()
    assert (staged / "images" / "camera.jpg").read_bytes() == (case.source / "images" / "camera.jpg").read_bytes()
    command, settings = case.submissions[0]
    assert command[:4] == ["/envs/gtsfm/bin/python", "-u", "-m", "gtsfm.runner"]
    for flag, value in {
        "--dataset_dir": ".gtsfm/staged/ray-flow/dataset_dir",
        "--num_workers": "3",
        "--threads_per_worker": "2",
        "--worker_memory_limit": "7GiB",
        "--correspondence_generator_config_name": "sift_brute_force",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert settings["num_gpus"] == 1 and settings["num_cpus"] == 6
    assert settings["cwd"] == "/opt/gtsfm"
    assert settings["submission_id"] == "gtsfm-ray-flow"
    assert "CUDA_VISIBLE_DEVICES" not in settings["env"]
    assert settings["env"]["GTSFM_LIVE_PREVIEW_INTERVAL"] == "150"
    output = Path(job.output_root)
    assert json.loads((output / "studio-inputs.json").read_text()) == {"dataset_dir": str(case.source)}
    assert yaml.safe_load((output / "config.yaml").read_text())["loader"]["dataset_dir"].startswith(".gtsfm/staged/")
    assert case.stops == ["gtsfm-ray-flow"]
    with TestClient(create_app(case.manager.results_root, job_manager=case.manager)) as client:
        scene = (output / "results" / "merged_retriangulated").relative_to(case.manager.results_root).as_posix()
        photos = client.get("/api/photos", params={"scene": scene})
        assert photos.status_code == 200
        thumbnail = client.get(photos.json()["items"][0]["thumbnail"])
        assert thumbnail.status_code == 200
        assert Image.open(BytesIO(thumbnail.content)).size == (32, 24)
        downloaded = client.get("/api/jobs/ray-flow/results")
        assert downloaded.status_code == 200
        with tarfile.open(fileobj=BytesIO(downloaded.content)) as archive:
            assert "results/merged_retriangulated/images.txt" in archive.getnames()
            assert archive.extractfile("studio-inputs.json").read() == (output / "studio-inputs.json").read_bytes()


def test_ray_uncertain_submission_is_not_replayed_and_owned_job_is_cleaned_up(ray_execution):
    """Retain one known Ray ID even if the submission response is lost."""
    case = ray_execution
    case.options["uncertain_submit"] = True
    job = _finish_ray_flow(case)
    assert job.status == "failed"
    assert "submission response lost" in job.error
    assert len(case.submissions) == 1
    assert case.stops == ["gtsfm-ray-flow"]
    assert not case.manager._ssh_workspaces
    assert job.ssh_spec is None


def test_ray_cancel_during_result_download_cannot_become_completed(ray_execution):
    """A completed remote process does not override cancellation during transfer."""
    case = ray_execution
    case.options["cancel_download"] = True
    job = _finish_ray_flow(case)
    assert job.status == "cancelled"
    assert not (Path(job.output_root) / "studio-inputs.json").exists()
    assert case.stops == ["gtsfm-ray-flow"]


def test_ray_stages_explicit_image_folder_and_preserves_its_local_mapping(ray_execution):
    """Separate image and dataset settings must both survive remote path translation."""
    case = ray_execution
    case.spec["images_dir"] = str(case.source / "images")
    job = _finish_ray_flow(case)
    assert job.status == "completed", job.error
    assert len(case.uploads) == 2
    command, _ = case.submissions[0]
    assert command[command.index("--images_dir") + 1] == ".gtsfm/staged/ray-flow/images_dir"
    assert (case.uploads[1][2] / "camera.jpg").is_file()
    mapping = json.loads((Path(job.output_root) / "studio-inputs.json").read_text())
    assert mapping["images_dir"] == str(case.source / "images")


def test_ray_rejects_cpu_reservations_larger_than_the_machine_before_submit(ray_execution):
    """Avoid an indefinitely pending Ray job for an oversubscribed Dask setting."""
    case = ray_execution
    case.spec["num_workers"] = 5
    job = _finish_ray_flow(case)
    assert job.status == "failed"
    assert "require 10 Ray CPUs" in job.error and "remote machine has 8" in job.error
    assert case.submissions == []


def test_ray_external_dask_workers_stage_constructor_and_reserve_only_driver_cpu(ray_execution):
    """External workers receive their resource settings without reserving head GPUs."""
    case = ray_execution
    case.spec.update(dask_worker_hosts="gpu-worker", num_workers=5)
    job = _finish_ray_flow(case)
    assert job.status == "completed", job.error
    command, settings = case.submissions[0]
    assert settings["num_cpus"] == 1 and settings["num_gpus"] == 0
    assert command[command.index("--cluster_config") + 1] == ".gtsfm/staged/ray-flow/cluster/cluster.yaml"
    constructor = yaml.safe_load((case.uploads[-1][2] / "cluster.yaml").read_text())
    assert constructor["_target_"] == "distributed.SSHCluster"
    assert constructor["remote_python"] == "/envs/gtsfm/bin/python"
    assert constructor["worker_options"]["n_workers"] == 5
    assert constructor["worker_options"]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
