"""Splat-only jobs reuse saved cameras, points, and correctly indexed original photos."""

import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visualization import runtime
from visualization.app import create_app, find_scenes
from visualization.reconstruction import point_count


@pytest.fixture
def saved_scene(tmp_path):
    root = tmp_path / "results"
    scene = root / "runs" / "example" / "results" / "merged_retriangulated"
    scene.mkdir(parents=True)
    images = scene / "images"
    images.mkdir()
    Image.new("RGB", (64, 48), (255, 0, 0)).save(images / "z.png")
    Image.new("RGB", (64, 48), (0, 255, 0)).save(images / "a.png")
    (scene / "cameras.txt").write_text("# cameras\n3 RADIAL 32 24 30 16 12 0 0\n")
    # Deliberately unsorted, non-contiguous image IDs. The reader sorts by name.
    (scene / "images.txt").write_text(
        "# images\n17 1 0 0 0 0 0 0 3 z.png\n16 12 5\n" "8 1 0 0 0 -1 0 0 3 a.png\n10 12 5\n"
    )
    (scene / "points3D.txt").write_text("# points\n5 0 0 5 128 128 0 0 17 0 8 0\n")
    return root, scene


def test_counts_hide_empty_scenes_and_refresh_after_write(saved_scene):
    root, scene = saved_scene
    empty = scene.parent / "ba_gt"
    empty.mkdir()
    (empty / "points3D.txt").write_text("# empty\n\n")
    (empty / "images.txt").write_text("# empty\n")
    items = find_scenes(root)
    assert len(items) == 1
    assert items[0]["point_count"] == 1
    assert items[0]["camera_count"] == 2
    assert items[0]["can_splatify"]
    (scene / "points3D.txt").write_text("# now empty\n")
    assert point_count(scene) == 0
    assert find_scenes(root) == []


def test_directory_modification_times_include_all_descendant_files(saved_scene):
    root, scene = saved_scene
    for path in root.rglob("*"):
        if path.is_file():
            os.utime(path, (100, 100))
    report = scene.parent / "reports" / "nested" / "metrics.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")
    os.utime(report, (300, 300))
    # Directory metadata and Studio's internal files do not date a reconstruction.
    os.utime(scene.parent, (900, 900))
    internal = root / ".gtsfm" / "job.log"
    internal.parent.mkdir()
    internal.write_text("internal")
    os.utime(internal, (1000, 1000))
    client = TestClient(create_app(root))
    response = client.get("/api/scenes").json()
    times = response["directory_modified_at"]
    assert times[scene.relative_to(root).as_posix()] == 100
    assert times[scene.parent.relative_to(root).as_posix()] == 300
    assert times["runs/example"] == times["runs"] == times["."] == 300
    assert ".gtsfm" not in times
    assert response["items"][0]["modified_at"] == 100
    # Updating an ordinary file changes every ancestor on the next discovery.
    os.utime(report, (400, 400))
    assert client.get("/api/scenes").json()["directory_modified_at"]["runs/example"] == 400


def test_empty_splat_files_are_hidden(saved_scene):
    root, scene = saved_scene
    splats = scene / "gaussian_splats.ply"
    splats.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n")
    assert find_scenes(root)[0]["splats"] is None
    (scene / "points3D.txt").write_text("# empty\n")
    assert find_scenes(root) == []
    splats.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 42\nend_header\n")
    items = find_scenes(root)
    assert len(items) == 1
    assert items[0]["kind"] == "splat"
    assert items[0]["splat_count"] == 42


@pytest.mark.parametrize("filename", ["points3D.txt", "images.txt", "cameras.txt", "gaussian_splats.ply"])
def test_result_revision_tracks_each_viewer_resource(saved_scene, filename):
    """Even same-size edits within one second must invalidate browser caches."""
    root, scene = saved_scene
    (scene / "gaussian_splats.ply").write_text("ply\nformat ascii 1.0\nelement vertex 42\nend_header\n")
    path = scene / filename
    os.utime(path, ns=(1_000_000_001, 1_000_000_001))
    before = find_scenes(root)[0]["revision"]
    os.utime(path, ns=(1_000_000_002, 1_000_000_002))
    after = find_scenes(root)[0]["revision"]
    assert before != after
    assert find_scenes(root)[0]["revision"] == after


def test_splat_launch_creates_separate_managed_job(saved_scene, monkeypatch):
    root, scene = saved_scene
    monkeypatch.setattr(runtime.JobManager, "_run", lambda *args: None)
    monkeypatch.setattr(
        runtime,
        "detect_hardware",
        lambda: {"devices": [{"id": "cuda:0", "status": "available", "supports_gaussian_splatting": True}]},
    )
    client = TestClient(create_app(root))
    response = client.post(
        "/api/scenes/splatify",
        json={
            "scene": scene.relative_to(root).as_posix(),
            "gs_max_steps": 123,
        },
    )
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["status"] == "queued"
    assert "gtsfm.splat.runner" in job["command"]
    assert "gtsfm.runner" not in job["command"]
    assert job["command"][-2:] == ["--max_steps", "123"]
    assert not Path(job["output_root"]).is_relative_to(scene)
    assert client.get(f'/api/jobs/{job["id"]}').status_code == 200
    assert (scene / "points3D.txt").is_file()


def test_retry_splat_preserves_source_and_training_settings(saved_scene, monkeypatch):
    root, scene = saved_scene
    monkeypatch.setattr(runtime.JobManager, "_run", lambda *args: None)
    monkeypatch.setattr(
        runtime,
        "detect_hardware",
        lambda: {"devices": [{"id": "cuda:0", "status": "available", "supports_gaussian_splatting": True}]},
    )
    manager = runtime.JobManager(root)
    client = TestClient(create_app(root, job_manager=manager))
    original = client.post(
        "/api/scenes/splatify",
        json={"scene": scene.relative_to(root).as_posix(), "gs_max_steps": 123},
    ).json()
    manager.get(original["id"]).status = "failed"
    response = client.post(f'/api/jobs/{original["id"]}/retry')
    assert response.status_code == 202, response.text
    retried = response.json()
    assert retried["spec"] == original["spec"]
    assert retried["output_root"] != original["output_root"]
    assert "gtsfm.splat.runner" in retried["command"]
    assert retried["command"][-2:] == ["--max_steps", "123"]
    assert str(scene.relative_to(root)) in retried["command"]
    assert manager.get(original["id"]).status == "failed"


@pytest.mark.parametrize("problem", ["empty", "missing_photo", "escape", "symlink", "preset", "steps"])
def test_invalid_splat_requests_do_not_start_jobs(saved_scene, monkeypatch, problem):
    root, scene = saved_scene
    launches = []
    monkeypatch.setattr(runtime.JobManager, "_run", lambda *args: launches.append(args))
    payload = {"scene": scene.relative_to(root).as_posix()}
    if problem == "empty":
        (scene / "points3D.txt").write_text("# no points\n")
    elif problem == "missing_photo":
        (scene / "images" / "a.png").unlink()
    elif problem == "escape":
        payload["scene"] = "../outside"
    elif problem == "symlink":
        outside = root.parent / "outside.txt"
        outside.write_text("5 0 0 5 128 128 0 0 17 0 8 0\n")
        (scene / "points3D.txt").unlink()
        (scene / "points3D.txt").symlink_to(outside)
    elif problem == "preset":
        payload["gaussian_splatting_config_name"] = "../../bad"
    else:
        payload["gs_max_steps"] = 0
    client = TestClient(create_app(root))
    assert client.post("/api/scenes/splatify", json=payload).status_code in {400, 422}
    assert client.get("/api/jobs").json()["items"] == []
    assert not launches


def test_splat_endpoint_requires_configured_api_key(saved_scene, monkeypatch):
    root, scene = saved_scene
    monkeypatch.setenv("GTSFM_API_KEY", "secret")
    client = TestClient(create_app(root))
    assert client.post("/api/scenes/splatify", json={"scene": scene.relative_to(root).as_posix()}).status_code == 401


def test_splat_import_keeps_camera_track_and_photo_indices_aligned(saved_scene):
    from gtsfm.splat.runner import load_reconstruction

    root, scene = saved_scene
    images, reconstruction = load_reconstruction(root, scene.relative_to(root).as_posix())
    assert len(images) == 2
    assert reconstruction.number_tracks() == 1
    # Image 8 (a.png) becomes index 0, image 17 (z.png) becomes index 1.
    assert reconstruction.get_image_info(0).name == "a.png"
    np.testing.assert_array_equal(images[0].value_array[0, 0], [0, 255, 0])
    np.testing.assert_array_equal(images[1].value_array[0, 0], [255, 0, 0])
    assert images[0].shape == (24, 32, 3)
    track = reconstruction.get_track(0)
    assert track.measurement(0)[0] == 1
    np.testing.assert_array_equal(track.measurement(0)[1], [16, 12])
    np.testing.assert_array_equal(reconstruction.get_camera(0).pose().translation(), [1, 0, 0])
