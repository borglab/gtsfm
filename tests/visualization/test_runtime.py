"""Tests for the installable browser workspace runtime."""

import json
import shlex
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visualization import app as app_runtime
from visualization import modal_deployment, runtime
from visualization import samples as sample_runtime
from visualization.app import create_app


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
    """Write one valid Gaussian-splat PLY without an external encoder."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property float f_dc_0",
                "property float f_dc_1",
                "property float f_dc_2",
                "property float opacity",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "property float rot_0",
                "property float rot_1",
                "property float rot_2",
                "property float rot_3",
                "end_header",
                "0 0 0 0 0 0 0 0 0 0 1 0 0 0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_configuration_schema_defaults_to_vggt() -> None:
    schema = runtime.configuration_schema()

    assert schema["defaults"]["config_name"] == "vggt"
    assert {item["id"] for item in schema["splat_implementations"]} == {"none", "gsplat", "anysplat"}
    assert "base_gs" in schema["gaussian_splatting_models"]
    models = {item["id"]: item for item in schema["models"]}
    assert models["vggt"]["capabilities"]["iterative_splat"] is True
    argoverse_options = {item["name"]: item for item in schema["loader_options"]["argoverse"]}
    assert argoverse_options["log_id"]["required"] is True


def test_configuration_catalog_detects_added_and_removed_presets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runtime, "CONFIG_ROOT", tmp_path)
    with TestClient(create_app(tmp_path / "results")) as client:
        assert client.get("/api/configuration").json()["models"] == []
        preset = tmp_path / "new_preset.yaml"
        preset.write_text("defaults: [vggt, _self_]\n", encoding="utf-8")
        models = client.get("/api/configuration").json()["models"]
        assert [model["id"] for model in models] == ["new_preset"]
        preset.unlink()
        assert client.get("/api/configuration").json()["models"] == []


def test_configuration_parameter_schema_follows_selected_hydra_preset() -> None:
    vggt = runtime.configuration_parameter_schema("vggt", "olsson", "base_gs", True)
    barn = runtime.configuration_parameter_schema("vggt_barn", "olsson", "base_gs", True)
    vggt_keys = {field["key"] for field in vggt["model"]["fields"]}
    barn_keys = {field["key"] for field in barn["model"]["fields"]}
    gaussian = {field["key"]: field for field in vggt["gaussian"]["fields"]}

    assert "cluster_optimizer.optimizer.geometry_transformer.config.confidence_threshold" in vggt_keys
    assert "cluster_optimizer.geometry_transformer.config.confidence_threshold" in barn_keys
    assert not any("_target_" in key or key.startswith("loader.") for key in vggt_keys)
    assert gaussian["init_type"]["type"] == "string"
    assert "max_steps" not in gaussian  # exposed as the prominent Training steps field


def test_typed_hydra_overrides_are_validated_for_selected_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setattr(runtime, "detect_hardware", _cuda_hardware)
    key = "cluster_optimizer.geometry_transformer.config.confidence_threshold"

    args, _ = runtime.build_runner_args(
        {
            "config_name": "vggt_barn",
            "loader": "olsson",
            "dataset_dir": str(dataset),
            "hardware": "cpu",
            "splat_implementation": "none",
            "hydra_overrides": {key: "6.5"},
        },
        tmp_path / "output",
    )

    assert f"{key}=6.5" in args
    with pytest.raises(ValueError, match="Unknown or protected Hydra settings"):
        runtime.build_runner_args(
            {
                "config_name": "vggt_barn",
                "loader": "olsson",
                "dataset_dir": str(dataset),
                "hardware": "cpu",
                "splat_implementation": "none",
                "hydra_overrides": {"cluster_optimizer.optimizer.not_in_this_preset": 1},
            },
            tmp_path / "bad-output",
        )


@pytest.mark.parametrize(
    "preset",
    [
        "fast_vggt",
        "fastvggt",
        "vggt",
        "vggt_barn",
        "vggt_cater",
        "vggt_church",
        "vggt_courthouse",
        "vggt_dub",
        "vggt_ig",
    ],
)
def test_fractional_reprojection_threshold_for_vggt_presets(
    preset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disabled float threshold must remain editable without relaxing integer fields."""

    schema = runtime.configuration_parameter_schema(preset, "olsson", "base_gs", False)
    fields = {field["key"].rsplit(".", 1)[-1]: field for field in schema["model"]["fields"]}
    threshold = fields["vggt_max_reproj_error"]
    count = fields["max_query_pts"]
    assert threshold["type"] == "number"
    assert threshold["defaultValue"] == 0.0
    assert threshold["step"] == "any"
    assert count["type"] == "integer"
    assert count["step"] == 1

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setattr(runtime, "detect_hardware", _cuda_hardware)
    spec = {
        "config_name": preset,
        "loader": "olsson",
        "dataset_dir": str(dataset),
        "hardware": "cpu",
        "splat_implementation": "none",
        "hydra_overrides": {threshold["key"]: "0.5", count["key"]: "17"},
    }
    args, _ = runtime.build_runner_args(spec, tmp_path / "output")
    assert f"{threshold['key']}=0.5" in args
    assert f"{count['key']}=17" in args

    # Form inputs reach the backend as strings; fractional counts must fail.
    with pytest.raises(ValueError, match="Invalid value for Hydra setting"):
        runtime.build_runner_args({**spec, "hydra_overrides": {count["key"]: "2.5"}}, tmp_path / "bad-output")


def test_expert_hydra_overrides_are_composed_before_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setattr(runtime, "detect_hardware", _cuda_hardware)
    spec = {
        "config_name": "vggt",
        "loader": "olsson",
        "dataset_dir": str(dataset),
        "hardware": "cpu",
        "splat_implementation": "none",
    }

    with pytest.raises(ValueError, match="Invalid Hydra override"):
        runtime.build_runner_args(
            {**spec, "advanced_overrides": ["cluster_optimizer.misspelled_setting=1"]},
            tmp_path / "bad-output",
        )
    with pytest.raises(ValueError, match="not editable"):
        runtime.build_runner_args(
            {**spec, "advanced_overrides": ["cluster_optimizer._target_=unsafe.Class"]},
            tmp_path / "protected-output",
        )


def test_yaml_catalog_ignores_cloud_duplicate_artifacts(tmp_path: Path) -> None:
    (tmp_path / "vggt.yaml").write_text("model: canonical\n", encoding="utf-8")
    (tmp_path / "vggt 2.yaml").write_text("model: duplicate\n", encoding="utf-8")
    (tmp_path / "_base.yaml").write_text("model: base\n", encoding="utf-8")

    assert runtime._yaml_stem_options(tmp_path) == ["vggt"]


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
            "gaussian_splatting_overrides": {
                "antialiased": True,
                "cull_alpha_thresh": 0.01,
            },
        },
        tmp_path / "output",
    )

    assert args[:2] == ["--config_name", "vggt"]
    assert "--run_gs" in args
    assert args[args.index("--gs_max_steps") + 1] == "123"
    gaussian_overrides = [
        args[index + 1] for index, value in enumerate(args) if value == "--gaussian_splatting_override"
    ]
    assert gaussian_overrides == [
        "+gaussian_splatting_optimizer.cfg.antialiased=true",
        "gaussian_splatting_optimizer.cfg.cull_alpha_thresh=0.01",
    ]
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_build_runner_args_rejects_unknown_gaussian_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setattr(runtime, "detect_hardware", _cuda_hardware)

    with pytest.raises(ValueError, match="Unknown Gaussian splatting settings"):
        runtime.build_runner_args(
            {
                "config_name": "vggt",
                "loader": "olsson",
                "dataset_dir": str(dataset),
                "hardware": "cuda:0",
                "splat_implementation": "gsplat",
                "gaussian_splatting_overrides": {"_target_": "unsafe"},
            },
            tmp_path / "output",
        )


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


def test_startup_assets_are_compressed_and_revalidated(tmp_path: Path, monkeypatch) -> None:
    static_root = tmp_path / "static"
    static_root.mkdir()
    asset = static_root / "studio.js"
    asset.write_text("// initial asset\n" * 300, encoding="utf-8")
    monkeypatch.setattr(app_runtime, "STATIC_ROOT", static_root)
    with TestClient(create_app(tmp_path / "results")) as client:
        page = client.get("/")
        assert page.headers["cache-control"] == "no-store"
        assert 'id="studio-startup"' in page.text
        assert "Opening your workspace" in page.text
        response = client.get("/static/studio.js", headers={"Accept-Encoding": "gzip"})
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["content-encoding"] == "gzip"
        assert "Accept-Encoding" in response.headers["vary"]
        assert response.text == asset.read_text(encoding="utf-8")
        etag = response.headers["etag"]
        cached = client.get("/static/studio.js", headers={"If-None-Match": etag})
        assert cached.status_code == 304
        assert not cached.content
        # A development edit must invalidate the cache immediately.
        asset.write_text("// changed asset, different length\n" * 300, encoding="utf-8")
        changed = client.get("/static/studio.js", headers={"If-None-Match": etag})
        assert changed.status_code == 200
        assert changed.headers["etag"] != etag
        assert changed.text == asset.read_text(encoding="utf-8")
        assert client.get("/api/scenes").headers["cache-control"] == "no-store"


def test_workspace_api_and_scene_discovery(tmp_path: Path) -> None:
    scene = tmp_path / "example" / "ba_output"
    scene.mkdir(parents=True)
    (scene / "points3D.txt").write_text("# points\n1 0 0 1 255 0 0 0\n", encoding="utf-8")
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
    parameters_response = client.get(
        "/api/configuration/parameters",
        params={"config_name": "vggt_barn", "loader": "olsson", "splat_implementation": "none"},
    )
    assert parameters_response.status_code == 200
    parameters = parameters_response.json()
    assert parameters["model"]["preset"] == "vggt_barn"
    assert parameters["model"]["fields"]
    assert parameters["gaussian"]["fields"] == []
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
    assert set(samples) == {
        "one-d-sfm",
        "argoverse",
        "astrovision-vesta",
        "lund-door",
        "crane-mast",
        "hilti-exp4",
        "imb-reichstag",
        "mobilebrick",
        "tanks-temples-barn",
    }
    assert samples["crane-mast"]["image_count"] == 2
    assert samples["crane-mast"]["recommendations"]["loader"] == "colmap"
    assert samples["lund-door"]["source_url"].startswith("https://github.com/borglab/gtsfm/")


def test_workspace_exports_saved_splat_as_ply(tmp_path: Path) -> None:
    source = tmp_path / "example" / "gaussian_splats.ply"
    _write_test_splat(source)
    client = TestClient(create_app(tmp_path))

    ply_response = client.get("/api/splats/export", params={"path": "example/gaussian_splats.ply", "format": "ply"})
    assert ply_response.status_code == 200
    assert '.ply"' in ply_response.headers["content-disposition"]
    assert ply_response.content.startswith(b"ply\n")


@pytest.mark.parametrize("saved_log", [True, False])
def test_failure_prompt_includes_command_and_full_log(tmp_path: Path, saved_log: bool) -> None:
    app = create_app(tmp_path)
    manager = app.state.gtsfm_jobs
    command = ["/env/bin/python", "-m", "gtsfm.runner", "--dataset_dir", "/data/Ravenna's photos"]
    job = runtime.ManagedJob(
        id="failed-job",
        name="Ravenna",
        status="failed",
        created_at="",
        updated_at="",
        output_root=str(tmp_path / "runs" / "Ravenna"),
        live_root="",
        spec={},
        command=command,
        return_code=1,
        error="Pipeline exited with code 1",
        log_tail=["ModuleNotFoundError: poselib"],
    )
    manager._jobs[job.id] = job
    full_log = "Early diagnostic\n" + "progress\n" * 300 + "```\nModuleNotFoundError: poselib\n"
    if saved_log:
        (manager.runtime_root / f"{job.id}.log").write_text(full_log, encoding="utf-8")
    response = TestClient(app).get(f"/api/jobs/{job.id}/diagnostic-prompt")
    assert response.status_code == 200
    prompt = response.json()["prompt"]
    assert "diagnose the root cause" in prompt
    assert shlex.join(command) in prompt
    assert "Exit code: 1" in prompt
    assert "ModuleNotFoundError: poselib" in prompt
    if saved_log:
        assert full_log in prompt
        assert "````text\n" in prompt
    else:
        assert "No saved log file" in prompt


def test_retry_preserves_settings_and_history_and_rebuilds_command(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runtime.JobManager, "_run", lambda *args: None)
    monkeypatch.setattr(runtime, "detect_hardware", _cuda_hardware)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    app = create_app(tmp_path / "results")
    manager = app.state.gtsfm_jobs
    client = TestClient(app)
    spec = {
        "name": "Ravenna",
        "dataset_dir": str(dataset),
        "loader": "olsson",
        "config_name": "vggt_sift_frontend_megaloc_phototourism_16gb",
        "hardware": "cuda:0",
        "splat_implementation": "none",
        "num_workers": 1,
        "threads_per_worker": 16,
        "worker_memory_limit": "23GB",
        "max_resolution": 760,
        "log": "DEBUG",
        "hydra_overrides": {"graph_partitioner.max_cameras": "18"},
        "advanced_overrides": ["graph_partitioner.min_cameras_to_partition=10"],
    }
    failed = manager.get(manager.start(spec)["id"])
    failed.status = "failed"
    failed.error = "Old failure"
    failed.return_code = 1
    failed.command = ["/old/environment/python", "stale-command"]
    old_log = manager.runtime_root / f"{failed.id}.log"
    old_log.write_text("original failure log", encoding="utf-8")
    original = failed.public()
    response = client.post(f"/api/jobs/{failed.id}/retry")
    assert response.status_code == 202
    retried = response.json()
    assert retried["id"] != failed.id
    assert retried["output_root"] != failed.output_root
    assert retried["spec"] == spec
    assert retried["status"] == "queued"
    assert retried["error"] is None and retried["return_code"] is None
    assert retried["command"][:3] == [runtime.sys.executable, "-m", "gtsfm.runner"]
    assert retried["command"][retried["command"].index("--threads_per_worker") + 1] == "16"
    assert retried["command"][retried["command"].index("--output_root") + 1] == retried["output_root"]
    assert "graph_partitioner.max_cameras=18" in retried["command"]
    assert "graph_partitioner.min_cameras_to_partition=10" in retried["command"]
    assert failed.public() == original
    assert old_log.read_text(encoding="utf-8") == "original failure log"
    # Duplicate clicks from different controls/tabs reuse the active retry.
    with ThreadPoolExecutor(max_workers=4) as pool:
        retries = list(pool.map(manager.retry, [failed.id] * 4))
    assert {job["id"] for job in retries} == {retried["id"]}
    assert len(manager.list()) == 2
    manager.get(retried["id"]).spec["hydra_overrides"]["graph_partitioner.max_cameras"] = "19"
    assert failed.spec["hydra_overrides"]["graph_partitioner.max_cameras"] == "18"
    # A new attempt is possible after the previous retry itself fails.
    manager.get(retried["id"]).status = "failed"
    assert manager.retry(failed.id)["id"] != retried["id"]


def test_retry_checks_auth_status_and_current_input_availability(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runtime.JobManager, "_run_remote", lambda *args: None)
    app = create_app(tmp_path)
    client = TestClient(app)
    manager = app.state.gtsfm_jobs
    queued = manager.start(
        {"execution_target": "remote", "remote_endpoint": "https://gpu.example", "api_key": "secret"}
    )
    monkeypatch.setenv("GTSFM_API_KEY", "workspace-secret")
    assert client.post(f"/api/jobs/{queued['id']}/retry").status_code == 401
    headers = {"Authorization": "Bearer workspace-secret"}
    assert client.post("/api/jobs/missing/retry", headers=headers).status_code == 404
    assert client.post(f"/api/jobs/{queued['id']}/retry", headers=headers).status_code == 400
    job = manager.get(queued["id"])
    job.status = "failed"
    job.remote = None
    job.spec = {"config_name": "vggt", "dataset_dir": str(tmp_path / "missing"), "splat_implementation": "none"}
    response = client.post(f"/api/jobs/{job.id}/retry", headers=headers)
    assert response.status_code == 400
    assert "does not exist" in response.json()["error"]
    assert len(manager.list()) == 1


def test_remote_retry_reuses_private_credentials_without_exposing_them(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runtime.JobManager, "_run_remote", lambda *args: None)
    manager = runtime.JobManager(tmp_path)
    original = manager.start({
        "name": "remote-retry", "execution_target": "remote", "remote_endpoint": "https://gpu.example",
        "api_key": "secret", "config_name": "vggt", "dataset_dir": "/data/original", "threads_per_worker": 16,
    })
    failed = manager.get(original["id"])
    failed.status = "failed"
    retried = manager.retry(failed.id)
    assert retried["spec"] == original["spec"]
    assert manager.get(retried["id"]).remote_api_key == "secret"
    assert "secret" not in json.dumps(retried)


def test_failure_prompt_checks_auth_and_job_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GTSFM_API_KEY", "test-key")
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/jobs/missing/diagnostic-prompt").status_code == 401
    assert client.get(
        "/api/jobs/missing/diagnostic-prompt", headers={"Authorization": "Bearer test-key"}
    ).status_code == 404


def test_remote_failure_prompt_uses_remote_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = runtime.JobManager(tmp_path)
    manager._jobs["remote-job"] = runtime.ManagedJob(
        id="remote-job",
        name="Remote",
        status="failed",
        created_at="",
        updated_at="",
        output_root="",
        live_root="",
        spec={},
        command=["remote", "https://gpu.example/api/jobs"],
        remote={"endpoint": "https://gpu.example", "job_id": "upstream"},
        remote_api_key="secret",
    )

    def remote_json(url, api_key):
        assert url == "https://gpu.example/api/jobs/upstream/diagnostic-prompt"
        assert api_key == "secret"
        return {"prompt": "Remote CLI and complete log"}

    monkeypatch.setattr(manager, "_remote_json", remote_json)
    assert manager.diagnostic_prompt("remote-job") == "Remote CLI and complete log"

    def unavailable(url, api_key):
        raise TimeoutError("Remote offline")

    monkeypatch.setattr(manager, "_remote_json", unavailable)
    prompt = manager.diagnostic_prompt("remote-job")
    assert "remote diagnostic report is unavailable" in prompt
    assert "secret" not in prompt


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
    live = client.get("/api/jobs/finished-job/live")
    assert live.status_code == 200
    assert live.json()["final_url"].endswith("/runs/finished/gaussian_splats.ply")
    response = client.get("/api/jobs/finished-job/splat", params={"format": "ply"})
    assert response.status_code == 200
    assert response.content.startswith(b"ply\n")


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
        log_tail=[
            "2026-08-13 17:00:00 [runner.py] INFO: 🌟 GTSFM: Starting SceneOptimizer...",
            "2026-08-13 17:00:01 [scene_optimizer.py] INFO: 🔥 GTSFM: Partitioning the view graph...",
        ],
    )

    response = TestClient(app).get("/api/jobs/running-job/live")

    assert response.status_code == 200
    assert response.json()["dask"]["workers"] == 2
    assert response.json()["dask"]["running_tasks"] == 3
    assert response.json()["message"] == "🔥 GTSFM: Partitioning the view graph..."


def test_workspace_rejects_invalid_splat_exports(tmp_path: Path) -> None:
    source = tmp_path / "gaussian_splats.ply"
    _write_test_splat(source)
    client = TestClient(create_app(tmp_path))

    unsupported = client.get("/api/splats/export", params={"path": source.name, "format": "obj"})
    assert unsupported.status_code == 400
    removed_spz = client.get("/api/splats/export", params={"path": source.name, "format": "spz"})
    assert removed_spz.status_code == 400
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


def test_remote_workspace_requests_use_certifi_ca_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_urlopen(_request: object, **kwargs: object) -> BytesIO:
        observed.update(kwargs)
        return BytesIO(b'{"status": "ready"}')

    monkeypatch.setattr(runtime.urllib.request, "urlopen", fake_urlopen)

    result = runtime.JobManager._remote_json("https://workspace.example/api/hardware", "secret")

    assert result == {"status": "ready"}
    assert observed["context"] is runtime._SSL_CONTEXT
    assert observed["timeout"] == runtime._REMOTE_REQUEST_TIMEOUT_SECONDS
    assert runtime._SSL_CONTEXT.get_ca_certs()


def test_remote_workspace_retries_truncated_status_response(monkeypatch: pytest.MonkeyPatch) -> None:
    responses: list[BytesIO] = [BytesIO(), BytesIO(b'{"status": "ready"}')]
    attempts = 0

    def fake_urlopen(_request: object, **_kwargs: object) -> BytesIO:
        nonlocal attempts
        response = responses[attempts]
        attempts += 1
        if attempts == 1:
            response.read = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
                runtime.http.client.IncompleteRead(b"", 12)
            )
        return response

    monkeypatch.setattr(runtime.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(runtime.time, "sleep", lambda _seconds: None)

    result = runtime.JobManager._remote_json("https://workspace.example/api/jobs/example", "secret")

    assert result == {"status": "ready"}
    assert attempts == 2


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
    # A fresh server discovers the upload without in-memory job state.
    items = TestClient(create_app(tmp_path)).get("/api/uploads").json()["items"]
    assert len(items) == 1
    assert items[0]["path"] == str(imported)
    assert items[0]["name"] == "example"
    assert items[0]["analysis"]["image_count"] == 1


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
    monkeypatch.setattr(threading.Thread, "start", lambda thread: thread.run())
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
    assert job.remote == {
        "endpoint": "https://gpu.example.test",
        "job_id": "upstream-job",
        "cancel_status": "confirmed",
    }
    assert job.log_tail[-1] == "Modal job cancellation confirmed."


def test_job_cancel_during_modal_start_forwards_after_remote_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = runtime.JobManager(tmp_path)
    job = runtime.ManagedJob(
        id="starting-remote-job",
        name="remote",
        status="running",
        created_at="now",
        updated_at="now",
        output_root=str(tmp_path / "output"),
        live_root=str(tmp_path / "live"),
        spec={"splat_implementation": "none"},
        command=[],
        remote={"endpoint": "https://gpu.example.test"},
        remote_api_key="secret",
    )
    manager._jobs[job.id] = job
    calls: list[tuple[str, object]] = []

    def fake_remote_json(
        endpoint: str, _api_key: str, *, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        calls.append((endpoint, payload))
        if endpoint.endswith("/api/jobs"):
            # Reproduce the user clicking Cancel while the POST is waiting for
            # Modal's cold-started container to accept the job.
            manager.cancel(job.id)
            return {"id": "upstream-job", "status": "queued"}
        return {"id": "upstream-job", "status": "cancelled"}

    monkeypatch.setattr(manager, "_remote_json", fake_remote_json)

    manager._run_remote(job)

    assert job.status == "cancelled"
    assert calls == [
        ("https://gpu.example.test/api/jobs", {"splat_implementation": "none", "execution_target": "local"}),
        ("https://gpu.example.test/api/jobs/upstream-job/cancel", {}),
    ]
    assert job.remote == {
        "endpoint": "https://gpu.example.test",
        "job_id": "upstream-job",
        "workspace_url": "https://gpu.example.test/?view=results",
        "cancel_status": "confirmed",
    }


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
        return {"path": "/mnt/gtsfm-studio/results/.gtsfm/uploads/remote-dataset"}

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

    assert posted["dataset_dir"] == "/mnt/gtsfm-studio/results/.gtsfm/uploads/remote-dataset"
    assert posted["execution_target"] == "local"
    assert managed.status == "completed"
    assert json.loads((Path(managed.live_root) / "status.json").read_text(encoding="utf-8")) == {}


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


def test_modal_workspace_api_key_is_stable_and_scoped() -> None:
    first = modal_deployment.modal_workspace_api_key("ak-one", "as-secret")

    assert first == modal_deployment.modal_workspace_api_key("ak-one", "as-secret")
    assert first != modal_deployment.modal_workspace_api_key("ak-two", "as-secret")
    assert "as-secret" not in first


def test_modal_deployment_rejects_invalid_machine_resources() -> None:
    manager = modal_deployment.ModalDeploymentManager(lambda *_args: {})

    with pytest.raises(ValueError, match="CPU allocation"):
        manager.start("ak-test", "as-test", "L40S", cpu=64, memory_mb=65536)
    with pytest.raises(ValueError, match="memory allocation"):
        manager.start("ak-test", "as-test", "L40S", cpu=8, memory_mb=1024)


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


def test_fastapi_uses_injected_hardware_provider_and_serves_favicon(tmp_path: Path) -> None:
    hardware = {
        "summary": "Remote GPU configured without allocation",
        "devices": [{"id": "cuda:0", "supports_gaussian_splatting": True}],
        "accelerator_count": 1,
        "platform": {},
    }
    client = TestClient(create_app(tmp_path, hardware_provider=lambda: hardware))

    assert client.get("/api/hardware").json() == hardware
    favicon = client.get("/favicon.ico")
    assert favicon.status_code == 200
    assert favicon.headers["content-type"] == "image/png"


def test_job_manager_accepts_safe_caller_provided_job_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(threading.Thread, "start", lambda _thread: None)
    manager = runtime.JobManager(tmp_path)
    spec = {
        "name": "durable-id",
        "dataset_dir": str(tmp_path),
        "loader": "olsson",
        "config_name": "vggt",
        "splat_implementation": "none",
        "hardware": "cpu",
    }

    job = manager.start(spec, job_id="modal_job-123")

    assert job["id"] == "modal_job-123"
    with pytest.raises(ValueError, match="unsupported characters"):
        manager.start(spec, job_id="../../unsafe")


def test_previous_uploads_find_legacy_folders_and_skip_unsafe_entries(tmp_path: Path) -> None:
    uploads = tmp_path / ".gtsfm" / "uploads"
    legacy = uploads / "123456abcdef" / "Ravenna71"
    legacy.mkdir(parents=True)
    Image.new("RGB", (12, 8)).save(legacy / "one.jpeg")
    (uploads / "abcdef123456").mkdir()  # Incomplete/empty upload.
    (uploads / "123456abcdef.tar").write_bytes(b"incomplete transfer")
    outside = tmp_path / "outside"
    outside.mkdir()
    Image.new("RGB", (12, 8)).save(outside / "private.jpg")
    (uploads / "111111111111").symlink_to(outside, target_is_directory=True)
    linked = uploads / "222222222222"
    linked.mkdir()
    (linked / "private.jpg").symlink_to(outside / "private.jpg")
    items = TestClient(create_app(tmp_path)).get("/api/uploads").json()["items"]
    assert len(items) == 1
    assert items[0]["name"] == "Ravenna71"
    assert items[0]["path"] == str(legacy)
    assert items[0]["file_count"] == 1
    assert items[0]["analysis"]["image_count"] == 1


def test_previous_uploads_preserve_archive_root_and_distinguish_duplicates(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    responses = [
        client.post(
            "/api/uploads/archive",
            files={"archive": ("dataset.tar", _tar_upload({"images/one.jpg": b"image"}), "application/x-tar")},
        ).json()
        for _ in range(2)
    ]
    items = TestClient(create_app(tmp_path)).get("/api/uploads").json()["items"]
    assert len(items) == 2
    assert {item["path"] for item in items} == {item["path"] for item in responses}
    assert len({item["id"] for item in items}) == 2
    assert all(item["name"] == "dataset" for item in items)
    assert items[0]["uploaded_at"] >= items[1]["uploaded_at"]


def test_previous_uploads_require_api_key_and_handle_empty_workspace(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/uploads").json() == {"items": []}
    monkeypatch.setenv("GTSFM_API_KEY", "secret")
    assert client.get("/api/uploads").status_code == 401
    assert client.get("/api/uploads", headers={"Authorization": "Bearer secret"}).status_code == 200
